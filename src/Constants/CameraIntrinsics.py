from Alt.Cameras.Parameters import CameraIntrinsics

class CameraIntrinsicsPredefined:
    #                       res             fov                     physical constants
    #   {CameraName} = ((HRes(pixels),Vres(pixels)),(Hfov(rad),Vfov(rad)),(Focal Length(mm),PixelSize(mm),sensor size(mm)), (CalibratedFx(pixels),CalibratedFy(pixels)),(CalibratedCx(pixels),CalibratedCy(pixels)))
    OV9782COLOR = CameraIntrinsics(
        640,
        480,  # Resolution
        1.22173,
        -1,  # FOV
        1.745,
        0.003,
        6.3,  # Physical Constants
        541.637,
        542.563,  # Calibrated Fx, Fy
        346.66661258567217,
        232.5032948773164,  # Calibrated Cx, Cy
    )

    SIMULATIONCOLOR = CameraIntrinsics(
        640,
        480,  # Resolution
        1.22173,
        0.9671,  # FOV
        1.745,
        0.003,
        6.3,  # Physical Constants
        604,
        414,  # Calibrated Fx, Fy
        320,
        240,  # Calibrated Cx, Cy
    )

    OAKESTIMATE = CameraIntrinsics(
        width_pix=1920,
        height_pix=1080,  # Resolution
        fx_pix=900,
        fy_pix=850,  # Calibrated Fx, Fy
        cx_pix=981,
        cy_pix=500,  # Calibrated Cx, Cy
    )
