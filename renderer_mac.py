import turtle
from math import sin, cos, tan, radians, sqrt
import time

import keyboard
from pynput.mouse import Listener as MouseListener

from vectors import *


class Camera:
    def __init__(self, pos, pitch, yaw):
        self.pos = pos
        self.pitch = pitch
        self.yaw = yaw

        turtle.tracer(0)
        turtle.setup(800, 600)
        pen = turtle.Turtle()
        pen.hideturtle()
        pen.penup()
        pen.color("black")
        pen.speed(0)
        self.pen = pen

    def move_axis(self, pos):
        self.pos += pos
        return self.pos

    def move(self, steps, horizontal_only=False):
        if horizontal_only:
            forward = Vector3(-sin(self.yaw), 0, cos(self.yaw))
        else:
            forward = self.get_view_direction()

        move = forward * steps
        self.pos += move
        return self.pos

    def strafe(self, steps):
        right = Vector3(cos(self.yaw), 0, sin(self.yaw))

        move = right * steps
        self.pos += move
        return self.pos

    def move_relative(self, pos, horizontal_only=False):
        self.move(pos.z, horizontal_only)
        self.strafe(pos.x)
        self.pos.y += pos.y

    def rotate(self, pitch, yaw):
        self.pitch += pitch
        self.yaw += yaw

        max_pitch = radians(89)
        self.pitch = max(-max_pitch, min(max_pitch, self.pitch))

    def get_view_direction(self):
        return Vector3(
            -sin(self.yaw) * cos(self.pitch),
            sin(self.pitch),
            cos(self.yaw) * cos(self.pitch),
        )

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

        aspect_ratio = screen_width / screen_height

        fov_rad = radians(fov)
        f = 1 / tan(fov_rad / 2)

        screen_x = (final.x * f / final.z) * (screen_height / screen_width)
        screen_y = final.y * f / final.z

        screen_x = screen_x * (screen_width / 2) + screen_width / 2
        screen_y = -screen_y * (screen_height / 2) + screen_height / 2

        turtle_x = screen_x - screen_width / 2
        turtle_y = screen_y - screen_height / 2

        return (turtle_x, turtle_y)

    def compute_face_normal(self, face):
        if len(face) < 3:
            return Vector3(0, 0, 1)

        edge1 = face[1] - face[0]
        edge2 = face[2] - face[0]
        normal = Vector3(
            edge1.y * edge2.z - edge1.z * edge2.y,
            edge1.z * edge2.x - edge1.x * edge2.z,
            edge1.x * edge2.y - edge1.y * edge2.x,
        ).normalize()

        return normal

    def is_face_visible(self, face, normal):
        if len(face) < 3:
            return True

        face_center = face[0]
        view_vector = (face_center - self.pos).normalize()
        dot_product = normal.dot(view_vector)

        return dot_product < 0

    def render(self, objects, materials, screen_width=800, screen_height=600, fov=90):
        rendered_faces = []
        for object in objects:
            for face, material in object["faces"]:
                normal = self.compute_face_normal(face)

                if not self.is_face_visible(face, normal):
                    continue

                projected = [
                    self.project_point(point, screen_width, screen_height, fov)
                    for point in face
                ]

                if not projected or None in projected:
                    continue

                centroid = sum(face, Vector3(0, 0, 0)) / len(face)

                diff = centroid - camera.pos
                distance = diff.dot(diff)

                color = materials[material]

                rendered_faces.append((projected, color, distance))

        rendered_faces.sort(key=lambda x: x[2], reverse=True)

        for projected, color, distance in rendered_faces:
            self.pen.goto(*projected[0])
            self.pen.fillcolor(color)
            self.pen.begin_fill()

            for pos in projected[1:]:
                self.pen.goto(*pos)

            self.pen.end_fill()


def rotate_point_around_axis(point, anchor, axis, angle):
    p = point - anchor
    k = axis.normalize()

    cos_a = cos(angle)
    sin_a = sin(angle)

    rotated = (
        p * cos_a
        + Vector3(k.y * p.z - k.z * p.y, k.z * p.x - k.x * p.z, k.x * p.y - k.y * p.x)
        * sin_a
        + k * (k.dot(p)) * (1 - cos_a)
    )

    return rotated + anchor


def rotate_objects(objects, anchor, rotation_vector):
    angle = sqrt(rotation_vector.x**2 + rotation_vector.y**2 + rotation_vector.z**2)

    if angle == 0:
        return

    axis = Vector3(
        rotation_vector.x / angle, rotation_vector.y / angle, rotation_vector.z / angle
    )

    for obj in objects:
        for face_idx, (face, material) in enumerate(obj["faces"]):
            rotated_face = []
            for vertex in face:
                rotated_vertex = rotate_point_around_axis(vertex, anchor, axis, angle)
                rotated_face.append(rotated_vertex)
            obj["faces"][face_idx] = (rotated_face, material)


def move_objects(objects, axis):
    for obj in objects:
        for face_idx, (face, material) in enumerate(obj["faces"]):
            for vertex in face:
                for i in range(len(face)):
                    face[i] = face[i] + axis


def load_obj(file_path):
    with open(file_path) as file:
        lines = file.readlines()

    objects = []
    vertexes = []
    curr_mtl = None

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        command, *args = parts

        if command == "o":
            name = args[0]
            objects.append({"name": name, "faces": []})
        elif command == "v":
            vertexes.append(Vector3(*map(float, args)))
        elif command == "f":
            objects[-1]["faces"].append(
                ([vertexes[int(i.split("/")[0]) - 1] for i in args], curr_mtl)
            )
        elif command == "usemtl":
            curr_mtl = args[0]

    return objects


def load_mtl(file_path):
    with open(file_path) as file:
        lines = file.readlines()

    colors = {}
    curr_mtl = None

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        command, *args = parts
        if command == "newmtl":
            curr_mtl = args[0]
        elif command == "Kd" and curr_mtl:
            colors[curr_mtl] = [float(i) for i in args]
    return colors


mouse_delta_x = 0
mouse_delta_y = 0
last_mouse_x = 0
last_mouse_y = 0
mouse_initialized = False


def on_mouse_move(x, y):
    global mouse_delta_x, mouse_delta_y, last_mouse_x, last_mouse_y, mouse_initialized, mouse_locked

    if not mouse_locked:
        return

    if not mouse_initialized:
        last_mouse_x = x
        last_mouse_y = y
        mouse_initialized = True
        return

    mouse_delta_x = x - last_mouse_x
    mouse_delta_y = y - last_mouse_y

    last_mouse_x = x
    last_mouse_y = y


def handle_movement(speed=0.2, sensitivity=0.05):
    global mouse_delta_x, mouse_delta_y

    camera_movement = zero3()
    camera_angle = zero2()

    if keyboard.is_pressed("w"):
        camera_movement.z = speed
    if keyboard.is_pressed("s"):
        camera_movement.z = -speed
    if keyboard.is_pressed("a"):
        camera_movement.x = -speed
    if keyboard.is_pressed("d"):
        camera_movement.x = speed
    if keyboard.is_pressed("ctrl"):
        camera_movement.y = speed
    if keyboard.is_pressed("space"):
        camera_movement.y = -speed

    if mouse_locked:
        camera_angle.x += mouse_delta_y * sensitivity
        camera_angle.y -= mouse_delta_x * sensitivity

        mouse_delta_x = 0
        mouse_delta_y = 0

    return camera_movement, camera_angle


def normalize_framerate(target):
    def decorator(func):
        def wrapped(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            time_to_sleep = max(0, (1 / (target + 2)) - elapsed)
            time.sleep(time_to_sleep)
            return result

        return wrapped

    return decorator


if __name__ == "__main__":
    turtle.bgcolor("#808080")
    camera = Camera(Vector3(20, -10, -20), radians(45), radians(45))
    cube = load_obj("rubic.obj")
    material = load_mtl("rubic.mtl")

    mouse_locked = True
    momentum = zero3()

    mouse_listener = MouseListener(on_move=on_mouse_move)
    mouse_listener.start()

    @normalize_framerate(60)
    def main():
        global mouse_locked, momentum

        camera.pen.clear()

        movement, angle = handle_movement(speed=0.03)
        momentum += movement
        camera.move_relative(momentum, horizontal_only=True)
        camera.rotate(*angle)
        momentum *= 0.9

        camera.render(cube, material, fov=103)

        turtle.update()

        if keyboard.is_pressed("p"):
            mouse_locked = not mouse_locked
            time.sleep(0.2)

        if keyboard.is_pressed("esc"):
            mouse_locked = False

    try:
        while True:
            main()
    except KeyboardInterrupt:
        mouse_listener.stop()
        print("Shutting down...")
