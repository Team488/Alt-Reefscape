import os

# For interactive display, do not force offscreen:
# os.environ["QT_QPA_PLATFORM"] = "offscreen"

import math
import json
import cv2
import numpy as np
import skfmm
import matplotlib

matplotlib.use("TkAgg")  # Use an interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap, Normalize

# --------------------- CUSTOM COLORMAP (as in your visualize method) ---------------------
custom_colors = [
    (1.0, 1.0, 0.8),  # light yellow
    (1.0, 0.9, 0.0),  # orange
    (1.0, 0.0, 0.0),  # red
    (0.5, 0.0, 0.5),  # purple
]
custom_cmap = LinearSegmentedColormap.from_list("heat_custom", custom_colors, N=256)
custom_cmap.set_bad("white")


# --------------------- PATH PLANNING FUNCTIONS ---------------------
# (For brevity, these functions are as you provided.)


class FastMarchingPathfinder:
    def __init__(self, grid_cost) -> None:
        self.grid_cost = grid_cost.copy()
        self.height, self.width = grid_cost.shape

    def compute_time_map(self, goal):
        speed = 1.0 / self.grid_cost
        phi = np.ones_like(self.grid_cost)
        goal_x, goal_y = goal
        phi[goal_y, goal_x] = -1
        time_map = skfmm.travel_time(phi, speed)
        return time_map

    def next_step(self, pos, time_map):
        x, y = pos
        best = pos
        best_time = time_map[y, x]
        for dx, dy in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if time_map[ny, nx] < best_time:
                    best_time = time_map[ny, nx]
                    best = (nx, ny)
        return best

    def bezier_curve(self, control_points, num_points=100):
        control_points = [np.array(pt, dtype=float) for pt in control_points]
        n = len(control_points)
        t_values = np.linspace(0, 1, num_points)
        if n == 2:
            curve = np.outer(1 - t_values, control_points[0]) + np.outer(
                t_values, control_points[1]
            )
        elif n == 3:
            p0, p1, p2 = control_points
            curve = (
                np.outer((1 - t_values) ** 2, p0)
                + np.outer(2 * (1 - t_values) * t_values, p1)
                + np.outer(t_values**2, p2)
            )
        elif n == 4:
            p0, p1, p2, p3 = control_points
            curve = (
                np.outer((1 - t_values) ** 3, p0)
                + np.outer(3 * t_values * (1 - t_values) ** 2, p1)
                + np.outer(3 * t_values**2 * (1 - t_values), p2)
                + np.outer(t_values**3, p3)
            )
        elif n == 5:
            p0, p1, p2, p3, p4 = control_points
            curve = (
                np.outer((1 - t_values) ** 4, p0)
                + np.outer(4 * t_values * (1 - t_values) ** 3, p1)
                + np.outer(6 * t_values**2 * (1 - t_values) ** 2, p2)
                + np.outer(4 * t_values**3 * (1 - t_values), p3)
                + np.outer(t_values**4, p4)
            )
        else:
            curve_points = []
            for tt in t_values:
                pts = control_points.copy()
                for r in range(1, n):
                    pts = [
                        (1 - tt) * pts[i] + tt * pts[i + 1] for i in range(len(pts) - 1)
                    ]
                curve_points.append(pts[0])
            curve = np.array(curve_points)
        return curve

    def check_collision(self, curve, minimum_heat=100):
        xs = np.rint(curve[:, 0]).astype(int)
        ys = np.rint(curve[:, 1]).astype(int)
        valid = (xs >= 0) & (xs < self.width) & (ys >= 0) & (ys < self.height)
        xs, ys = xs[valid], ys[valid]
        return np.any(self.grid_cost[ys, xs] >= minimum_heat)

    def try_inflate_segment(self, segment, max_offset_pixels=200, tol=1.0):
        if len(segment) < 2:
            return None
        p0 = np.array(segment[0], dtype=float)
        p_end = np.array(segment[-1], dtype=float)
        chord = p_end - p0
        chord_length = np.linalg.norm(chord)
        if chord_length == 0:
            return None
        perp = np.array([-chord[1], chord[0]]) / chord_length
        candidate_list = []
        for sign in [1, -1]:
            lower, upper = 0, max_offset_pixels
            best_candidate_dir = None
            while upper - lower > tol:
                mid_offset = (lower + upper) / 2.0
                mid = (p0 + p_end) / 2 + sign * perp * mid_offset
                candidate_segment = [segment[0], tuple(mid), segment[-1]]
                candidate_curve = self.bezier_curve(candidate_segment, num_points=100)
                collision = self.check_collision(candidate_curve)
                if not collision:
                    best_candidate_dir = candidate_segment
                    upper = mid_offset
                else:
                    lower = mid_offset
            if best_candidate_dir is not None:
                candidate_list.append(best_candidate_dir)
        if candidate_list:

            def candidate_offset(candidate):
                mid = np.array(candidate[1])
                return np.linalg.norm(mid - (p0 + p_end) / 2)

            best_candidate = min(candidate_list, key=candidate_offset)
            return best_candidate
        return None

    def generate_safe_bezier_paths(self, control_points):
        segments = []
        segment = [control_points[0]]
        for i in range(1, len(control_points)):
            segment.append(control_points[i])
            curve = self.bezier_curve(segment, num_points=100)
            if self.check_collision(curve):
                inflated_segment = self.try_inflate_segment(segment)
                if inflated_segment is not None:
                    segment = inflated_segment
                    curve = self.bezier_curve(segment, num_points=100)
                    if self.check_collision(curve):
                        segments.append(segment[:-1])
                        segment = [control_points[i - 1], control_points[i]]
                else:
                    segments.append(segment[:-1])
                    segment = [control_points[i - 1], control_points[i]]
        segments.append(segment)
        return [np.array(seg) for seg in segments]


def deflate_inflection_points(points, distance_threshold=2):
    if not points:
        return []
    group = [np.array(points[0], dtype=float)]
    deflated_points = []
    for pt in points[1:]:
        pt = np.array(pt, dtype=float)
        if np.linalg.norm(pt - group[-1]) <= distance_threshold:
            group.append(pt)
        else:
            avg_point = np.mean(group, axis=0)
            deflated_points.append(tuple(avg_point))
            group = [pt]
    if group:
        avg_point = np.mean(group, axis=0)
        deflated_points.append(tuple(avg_point))
    return deflated_points


def get_static_obstacles(filename):
    try:
        with open(filename, "r") as f:
            obstacles = json.load(f)
        return obstacles
    except Exception:
        return [[150, 100], [300, 150], [450, 200]]


def apply_and_inflate_all_static_obstacles(grid, static_obs_array, safe_distance):
    for coord in static_obs_array:
        x, y = coord
        if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
            grid[y, x] = 1000000
    binary_static = (grid > 1).astype(np.uint8)
    kernel_size = int(2 * safe_distance)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    inflated_static = cv2.dilate(binary_static, kernel, iterations=1)
    grid[inflated_static == 1] = 101
    return grid


def find_inflection_points(path):
    if len(path) < 3:
        return path
    inflection_points = [path[0]]
    for i in range(1, len(path) - 1):
        prev_dx = path[i][0] - path[i - 1][0]
        prev_dy = path[i][1] - path[i - 1][1]
        next_dx = path[i + 1][0] - path[i][0]
        next_dy = path[i + 1][1] - path[i][1]
        if (prev_dx, prev_dy) != (next_dx, next_dy):
            inflection_points.append(path[i])
    inflection_points.append(path[-1])
    return inflection_points


def filter_path_until_first_non_heat(path, grid, threshold=100):
    for i, pt in enumerate(path):
        x = int(round(pt[0]))
        y = int(round(pt[1]))
        if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
            if grid[y, x] < threshold:
                return path[i:]
    return []


# --------------------- CONSTANTS ---------------------
fieldHeightMeters = 8.05
fieldWidthMeters = 17.55
grid_width = 690
grid_height = 316
ROBOT_SIZE_LENGTH_INCHES = 36
ROBOT_SIZE_WIDTH_INCHES = 35
DEFAULT_SAFE_DISTANCE_INCHES = 5

MAX_ROBOT_SIZE_DIAGONAL_INCHES = math.sqrt(
    ROBOT_SIZE_LENGTH_INCHES**2 + ROBOT_SIZE_WIDTH_INCHES**2
)
CENTER_ROBOT_SIZE = MAX_ROBOT_SIZE_DIAGONAL_INCHES / 2
PIXELS_PER_METER_X = grid_width / fieldWidthMeters
PIXELS_PER_METER_Y = grid_height / fieldHeightMeters

static_obs_array = get_static_obstacles("../pathplanning/static_obstacles.json")
static_hang_obs_red_far = []
static_hang_obs_red_mid = []
static_hang_obs_red_close = []
static_hang_obs_blue_far = []
static_hang_obs_blue_mid = []
static_hang_obs_blue_close = []


# --------------------- Dummy Request & Helper Classes ---------------------
class DummyPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class DummyOptions:
    pass


class DummyRequest:
    def __init__(self, start, end, safeDistanceInches):
        self.start = start
        self.end = end
        self.safeDistanceInches = safeDistanceInches

    def HasField(self, field):
        return True

    options = DummyOptions()


# --------------------- PATH PLANNING (Simulated Request) ---------------------
def run_pathplanning(request):
    base_grid = np.ones((grid_height, grid_width), dtype=float)
    start = (request.start.x, request.start.y)
    goal = (request.end.x, request.end.y)
    SAFE_DISTANCE_INCHES = (
        request.safeDistanceInches
        if request.HasField("safeDistanceInches")
        else DEFAULT_SAFE_DISTANCE_INCHES
    )
    TOTAL_SAFE_DISTANCE = int(CENTER_ROBOT_SIZE + SAFE_DISTANCE_INCHES)
    modified_static_obs = static_obs_array
    static_grid = apply_and_inflate_all_static_obstacles(
        base_grid, modified_static_obs, TOTAL_SAFE_DISTANCE
    )
    pathfinder = FastMarchingPathfinder(static_grid)
    START = (int(start[0] * PIXELS_PER_METER_X), int(start[1] * PIXELS_PER_METER_Y))
    GOAL = (int(goal[0] * PIXELS_PER_METER_X), int(goal[1] * PIXELS_PER_METER_Y))
    time_map = pathfinder.compute_time_map(GOAL)
    path = [START]
    current = START
    max_steps = 10000
    for _ in range(max_steps):
        next_cell = pathfinder.next_step(current, time_map)
        if next_cell == current:
            break
        path.append(next_cell)
        current = next_cell
        if current == GOAL:
            break
    path = filter_path_until_first_non_heat(path, static_grid)
    inflection_points = find_inflection_points(path)
    if SAFE_DISTANCE_INCHES >= 10:
        smoothed_control_points = deflate_inflection_points(
            inflection_points, distance_threshold=4
        )
    else:
        smoothed_control_points = deflate_inflection_points(inflection_points)
    safe_bezier_segments = pathfinder.generate_safe_bezier_paths(
        smoothed_control_points
    )
    safe_segment_curves = [
        pathfinder.bezier_curve(seg, num_points=100) for seg in safe_bezier_segments
    ]
    return {
        "static_grid": static_grid,
        "time_map": time_map,
        "fast_path": np.array(path),
        "inflection_points": np.array(inflection_points),
        "safe_segments": safe_bezier_segments,
        "safe_segment_curves": safe_segment_curves,
        "pathfinder": pathfinder,
    }


# --------------------- ANIMATION SETUP ---------------------
fps = 120
interval = 1000 / fps

dummy_request = DummyRequest(
    start=DummyPoint(2, 5), end=DummyPoint(12.57, 2), safeDistanceInches=0
)
result = run_pathplanning(dummy_request)

static_grid = result["static_grid"]
time_map = result["time_map"]
fast_path = result["fast_path"]
inflection_points = result["inflection_points"]
safe_segments = result["safe_segments"]
safe_segment_curves = result["safe_segment_curves"]
pathfinder = result["pathfinder"]

path_frames = len(fast_path)
inflection_phase = 30
seg_frames_per_segment = 50
num_seg = len(safe_segment_curves)
total_seg_frames = num_seg * seg_frames_per_segment
total_frames = path_frames + inflection_phase + total_seg_frames

# Build display_map as in your static visualize() method.
display_map = np.where(static_grid <= 1, np.nan, time_map)
extent = [0, static_grid.shape[1], 0, static_grid.shape[0]]
mask = static_grid > 1
if np.any(mask):
    vmin = np.nanmin(display_map[mask])
    vmax = np.nanmax(display_map[mask])
else:
    vmin, vmax = 0, 1

fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
ax.set_title("Probability Heatmap Pathplanner")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])
im = ax.imshow(
    display_map, cmap=custom_cmap, extent=extent, origin="lower", vmin=vmin, vmax=vmax
)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Obstacle Heat")
ax.grid(True, color="gray", linestyle="--", linewidth=0.5)

# Plot start and goal (static)
xs, ys = zip(*fast_path)
start_sc = ax.scatter(
    xs[0], ys[0], color="blue", edgecolors="black", s=100, label="Start"
)
goal_sc = ax.scatter(
    xs[-1], ys[-1], color="cyan", edgecolors="black", s=100, label="Goal"
)

# Initialize artists for the animated discrete path, inflection points, safe Bézier curves, and inflation cords.
(disc_path_line,) = ax.plot([], [], color="red", linewidth=2, label="Discrete Path")
(inflection_line,) = ax.plot([], [], "ro--", label="Inflection Points")

# Create a list for safe Bézier curves.
bez_line_list = []
for i, _ in enumerate(safe_segment_curves):
    label = "Safe Bézier Curve" if i == 0 else None
    (ln,) = ax.plot([], [], "b-", linewidth=2, label=label)
    bez_line_list.append(ln)

# Create a list for inflation cords (chord lines).
chord_line_list = []
control_scatter = []

for i, seg in enumerate(safe_segments):
    label = "Inflation Cord" if i == 0 else None
    (cln,) = ax.plot([], [], "k--", linewidth=1, label=label)
    chord_line_list.append(cln)
    cscatter = ax.scatter([], [], c="purple", s=40, zorder=6)
    control_scatter.append(cscatter)

ax.legend(loc="upper right")


def update(frame):
    # Phase 1: Animate the discrete path.
    print(frame)
    if frame < path_frames:
        current_path = fast_path[: frame + 1]
        disc_path_line.set_data(current_path[:, 0], current_path[:, 1])
    else:
        disc_path_line.set_data(fast_path[:, 0], fast_path[:, 1])
    # Phase 2: Display inflection points after the discrete path is complete.
    if frame >= path_frames and frame < path_frames + inflection_phase:
        inflection_line.set_data(inflection_points[:, 0], inflection_points[:, 1])
    elif frame >= path_frames + inflection_phase:
        inflection_line.set_data(inflection_points[:, 0], inflection_points[:, 1])
    # Phase 3: Animate safe Bézier curves sequentially.
    seg_anim_frame = frame - (path_frames + inflection_phase)
    seg_index = seg_anim_frame // seg_frames_per_segment
    sub_frame = seg_anim_frame % seg_frames_per_segment
    for i in range(num_seg):
        if i < seg_index:
            curve = safe_segment_curves[i]
            bez_line_list[i].set_data(curve[:, 0], curve[:, 1])
            control_scatter[i].set_offsets(np.array(safe_segments[i]))
        elif i == seg_index and seg_index < num_seg:
            curve = safe_segment_curves[i]
            num_pts = len(curve)
            pts_to_show = int((sub_frame / seg_frames_per_segment) * num_pts)
            pts_to_show = max(2, min(pts_to_show, num_pts))
            bez_line_list[i].set_data(curve[:pts_to_show, 0], curve[:pts_to_show, 1])
            control_scatter[i].set_offsets(np.array(safe_segments[i]))

        else:
            bez_line_list[i].set_data([], [])

    # Phase 4: Animate the inflation cords (chord lines).
    for i in range(num_seg):
        # WORKING (draws the full control polygon)
        if i <= seg_index:
            ctrl = np.array(safe_segments[i])
            chord_line_list[i].set_data(ctrl[:, 0], ctrl[:, 1])
        else:
            chord_line_list[i].set_data([], [])

    return (
        disc_path_line,
        inflection_line,
        *bez_line_list,
        *chord_line_list,
        *control_scatter,
    )


# Create the animation
ani = FuncAnimation(fig, update, frames=int(total_frames), interval=interval, blit=True)
print("saving gif...")
ani.save("pathplanning120fps.gif", writer="pillow", fps=120)
print("gif done")

# Update the figure to the final frame manually
update(total_frames - 1)
plt.savefig("pathplanning120fps.png")
plt.close(fig)
