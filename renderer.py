import turtle
from math import sin, cos, tan, radians
import time

from vectors import Vector2, Vector3


class Camera:
    def __init__(self, pos, pitch, yaw):
        self.pos = pos
        self.pitch = pitch
        self.yaw = yaw

    def move_axis(self, pos):
        self.pos += pos
        return self.pos

    def move(self, steps):
        pitch = self.pitch
        yaw = self.yaw

        move = Vector3(
            cos(pitch) * cos(yaw) * steps,
            sin(pitch) * steps,
            cos(pitch) * sin(yaw) * steps,
        )
        self.pos += move
        return self.pos

    def rotate_camera(self, pitch, yaw):
        self.pitch += pitch
        self.yaw += yaw

    def project_point(self, point, screen_width=800, screen_height=600, fov=90):
        relative = point - self.pos

        yaw = self.yaw
        pitch = self.pitch

        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)
        cos_pitch = cos(pitch)
        sin_pitch = sin(pitch)

        xz = Vector3(
            cos_yaw * relative.x + sin_yaw * relative.z,
            relative.y,
            -sin_yaw * relative.x + cos_yaw * relative.z,
        )

        final = Vector3(
            xz.x,
            cos_pitch * xz.y - sin_pitch * xz.z,
            sin_pitch * xz.y + cos_pitch * xz.z,
        )

        if final.z <= 0:
            return None

        f = screen_width / (2 * tan(radians(fov) / 2))
        screen_x = (final.x * f / final.z) + screen_width / 2
        screen_y = (final.y * f / final.z) + screen_height / 2

        return (screen_x, screen_y)


def draw_line(pen, camera, point1, point2):
    p1 = camera.project_point(point1)
    p2 = camera.project_point(point2)

    if p1 and p2:
        x1, y1 = p1[0] - 400, p1[1] - 300
        x2, y2 = p2[0] - 400, p2[1] - 300
        pen.penup()
        pen.goto(x1, y1)
        pen.pendown()
        pen.goto(x2, y2)
        pen.penup()


if __name__ == "__main__":
    camera = Camera(Vector3(4, 4, -5), radians(-45), radians(45))

    cube_corners = [
        Vector3(-1, -1, -1),
        Vector3(-1, -1, 1),
        Vector3(-1, 1, -1),
        Vector3(-1, 1, 1),
        Vector3(1, -1, -1),
        Vector3(1, -1, 1),
        Vector3(1, 1, -1),
        Vector3(1, 1, 1),
    ]

    edges = [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ]

    turtle.tracer(0)
    pen = turtle.Turtle()
    pen.hideturtle()
    pen.penup()
    pen.color("black")

    for edge in edges:
        draw_line(pen, camera, cube_corners[edge[0]], cube_corners[edge[1]])

    turtle.update()
    turtle.done()
