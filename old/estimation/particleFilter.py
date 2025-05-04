import numpy as np
from tools.Constants import MapConstants
from Core.ConfigOperator import staticLoad


class ParticleFilter:
    def __init__(self, numParticles=1000) -> None:
        self.obstacles, _ = staticLoad("obstacleMap.npy")
        self.NUM_PARTICLES = numParticles
        self.STATE_DIM = 4  # [x, y, vx, vy]
        self.field_bounds = [
            0,
            MapConstants.fieldWidth.getCM(),
            0,
            MapConstants.fieldHeight.getCM(),
        ]

        # Initialize particles randomly within bounds and velocities within [-1, 1]
        self.particles = np.random.uniform(
            low=[self.field_bounds[0], self.field_bounds[2], -1, -1],
            high=[self.field_bounds[1], self.field_bounds[3], 1, 1],
            size=(self.NUM_PARTICLES, self.STATE_DIM),
        )

        self.weights = np.ones(self.NUM_PARTICLES) / self.NUM_PARTICLES

    def state_transition(self, dt=1) -> None:
        max_speed = 10000
        max_accel = 10000

        # Add position noise for more spread
        pos_noise = np.random.normal(
            0, sum(self.field_bounds) / 500, size=(self.NUM_PARTICLES, 2)
        )

        accel_noise = np.random.normal(
            0, sum(self.field_bounds) / 1400, size=(self.NUM_PARTICLES, 2)
        )  # Slightly larger noise
        accel_noise = np.clip(accel_noise, -max_accel, max_accel)
        self.particles[:, 2:] += accel_noise

        speeds = np.linalg.norm(self.particles[:, 2:], axis=1)
        mask = speeds > max_speed
        self.particles[mask, 2:] *= (max_speed / speeds[mask])[:, None]

        # Update positions with velocity and noise
        self.particles[:, :2] += self.particles[:, 2:] * dt + pos_noise

    def apply_constraints(self, currentRobotHeight=0) -> None:
        for i, particle in enumerate(self.particles):
            x, y, vx, vy = particle

            # Reflective boundaries
            if x < self.field_bounds[0] or x > self.field_bounds[1]:
                self.particles[i, 0] = np.clip(
                    x, self.field_bounds[0], self.field_bounds[1] - 1
                )
                self.particles[i, 2] *= -1

            if y < self.field_bounds[2] or y > self.field_bounds[3]:
                self.particles[i, 1] = np.clip(
                    y, self.field_bounds[2], self.field_bounds[3] - 1
                )
                self.particles[i, 3] *= -1

            # Obstacle penalty if robot does not fit under
            nx, ny = self.particles[i][:2]
            if self.obstacles[int(ny), int(nx)] <= currentRobotHeight:
                self.weights[i] = 0

        # Normalize weights to prevent all-zero issue
        # self.weights += 1e-300
        self.weights /= np.sum(self.weights)

    def update_weights(self, observationPosition, observationVelocity) -> None:
        print(len(self.particles))
        # Normalize state and observation values by field dimensions
        norm_positions = self.particles[:, :2] / [
            self.field_bounds[1],
            self.field_bounds[3],
        ]
        norm_obs_position = observationPosition / [
            self.field_bounds[1],
            self.field_bounds[3],
        ]

        # Compute normalized errors
        pos_error = np.linalg.norm(norm_positions - norm_obs_position, axis=1)
        vel_error = np.linalg.norm(self.particles[:, 2:] - observationVelocity, axis=1)

        # Adjust exponential decay to be less aggressive
        pos_weight = np.exp(-pos_error / 0.01)  # Adjust scaling factor
        vel_weight = np.exp(-vel_error / 0.01)  # Adjust scaling factor

        self.weights = pos_weight + vel_weight
        self.weights += 1e-100  # Prevent zero weights
        self.weights /= np.sum(self.weights)

    def resample(self):
        indices = np.random.choice(
            self.NUM_PARTICLES, size=self.NUM_PARTICLES, p=self.weights
        )
        self.particles = self.particles[indices]

        # Compute spread of particles
        pos_variance = np.var(self.particles[:, :2], axis=0)
        vel_variance = np.var(self.particles[:, 2:], axis=0)

        # Add adaptive noise based on spread
        noise_scale = np.hstack([np.sqrt(pos_variance), np.sqrt(vel_variance)])  # (4,)
        noise = np.random.normal(0, noise_scale / 100, size=self.particles.shape)
        self.particles += noise

        self.weights.fill(1.0 / self.NUM_PARTICLES)  # Reset weights
        return self.particles, self.weights
