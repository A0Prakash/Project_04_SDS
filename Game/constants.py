

# Colors
FIELD_COLOR = (255, 255, 255)
BOUNDARY_COLOR = (200, 200, 200)
BOUNDARY_THICKNESS = 4

# Real field dimensions in inches (10ft x 15ft)
FIELD_WIDTH_INCHES = 15*12  # 15ft * 12 inches
FIELD_HEIGHT_INCHES = 22*12  # 10ft * 12 inches

# Scale everything by 2 for pixels
PIXELS_PER_INCH = 3

# Window and Field Settings
FIELD_MARGIN = 20
FIELD_WIDTH = FIELD_WIDTH_INCHES * PIXELS_PER_INCH  # 180 * 2 = 360 pixels
FIELD_HEIGHT = FIELD_HEIGHT_INCHES * PIXELS_PER_INCH  # 120 * 2 = 240 pixels
WINDOW_WIDTH = FIELD_WIDTH + (2 * FIELD_MARGIN)
WINDOW_HEIGHT = FIELD_HEIGHT + (2 * FIELD_MARGIN)

# Robot will be 30 inches * 2 = 60 pixels
ROBOT_SIZE_PIXELS = 30 * PIXELS_PER_INCH  # 60 pixels

# For Box2D we'll still need meters conversion
INCHES_TO_METERS = 0.0254
PPM = PIXELS_PER_INCH / INCHES_TO_METERS  # pixels per meter

GAME_DURATION = 15  # seconds
BLUE_START_Y = WINDOW_HEIGHT - FIELD_MARGIN - 100  # 100 pixels from bottom
RED_START_Y = FIELD_MARGIN + 100  # 100 pixels from top

GOAL_HEIGHT = 60  # Height of the goal zone in pixels
STRIPE_WIDTH = 20  # Width of each diagonal stripe
GOAL_COLOR_1 = (220, 220, 220)  # Light gray
GOAL_COLOR_2 = (200, 200, 200)  # Slightly darker gray

