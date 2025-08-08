import turtle
from math import *
import time
from typing import Union, List, Dict, Tuple
import keyboard
import sys
import os

try:
    using_mouse = True
    import mouse
except (ImportError, OSError):
    using_mouse = False
    from pynput.mouse import Listener as MouseListener


Number = Union[int, float]


# Vectors
class Vector2:
    def __init__(self, x: Number, y: Number):
        self.x, self.y = float(x), float(y)

    def __repr__(self):
        return f"Vector2({self.x}, {self.y})"

    def __iter__(self):
        yield self.x
        yield self.y

    def __add__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: Number) -> "Vector2":
        return Vector2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: Number) -> "Vector2":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Number) -> "Vector2":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector2(self.x / scalar, self.y / scalar)

    def __floordiv__(self, scalar: Number) -> "Vector2":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector2(self.x // scalar, self.y // scalar)

    def __neg__(self) -> "Vector2":
        return Vector2(-self.x, -self.y)

    def __eq__(self, other: "Vector2") -> bool:
        # Use small epsilon for floating point comparison
        epsilon = 1e-10
        return abs(self.x - other.x) < epsilon and abs(self.y - other.y) < epsilon

    def __hash__(self):
        return hash((round(self.x, 10), round(self.y, 10)))

    def dot(self, other: "Vector2") -> float:
        return self.x * other.x + self.y * other.y

    def magnitude(self) -> float:
        return sqrt(self.x * self.x + self.y * self.y)

    def magnitude_squared(self) -> float:
        return self.x * self.x + self.y * self.y

    def normalize(self) -> "Vector2":
        mag = self.magnitude()
        if mag == 0:
            return Vector2(0, 0)
        return Vector2(self.x / mag, self.y / mag)

    def distance_to(self, other: "Vector2") -> float:
        return (self - other).magnitude()

    def angle(self) -> float:
        return atan2(self.y, self.x)

    def rotate(self, angle: float) -> "Vector2":
        cos_a, sin_a = cos(angle), sin(angle)
        return Vector2(self.x * cos_a - self.y * sin_a, self.x * sin_a + self.y * cos_a)

    def is_zero(self) -> bool:
        epsilon = 1e-10
        return abs(self.x) < epsilon and abs(self.y) < epsilon


class Vector3:
    def __init__(self, x: Number, y: Number, z: Number):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __repr__(self):
        return f"Vector3({self.x}, {self.y}, {self.z})"

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: Number) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: Number) -> "Vector3":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Number) -> "Vector3":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __floordiv__(self, scalar: Number) -> "Vector3":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector3(self.x // scalar, self.y // scalar, self.z // scalar)

    def __neg__(self) -> "Vector3":
        return Vector3(-self.x, -self.y, -self.z)

    def __eq__(self, other: "Vector3") -> bool:
        epsilon = 1e-10
        return (
            abs(self.x - other.x) < epsilon
            and abs(self.y - other.y) < epsilon
            and abs(self.z - other.z) < epsilon
        )

    def __hash__(self):
        return hash((round(self.x, 10), round(self.y, 10), round(self.z, 10)))

    def dot(self, other: "Vector3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def magnitude(self) -> float:
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def magnitude_squared(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def normalize(self) -> "Vector3":
        mag = self.magnitude()
        if mag == 0:
            return Vector3(0, 0, 0)
        return Vector3(self.x / mag, self.y / mag, self.z / mag)

    def distance_to(self, other: "Vector3") -> float:
        return (self - other).magnitude()

    def is_zero(self) -> bool:
        epsilon = 1e-10
        return abs(self.x) < epsilon and abs(self.y) < epsilon and abs(self.z) < epsilon

    def project_onto(self, other: "Vector3") -> "Vector3":
        if other.is_zero():
            return Vector3(0, 0, 0)
        return other * (self.dot(other) / other.magnitude_squared())

    def reflect(self, normal: "Vector3") -> "Vector3":
        return self - 2 * self.project_onto(normal)

    def to_radians(self):
        return Vector3(*[radians(i) for i in self])


def zero2() -> Vector2:
    return Vector2(0, 0)


def zero3() -> Vector3:
    return Vector3(0, 0, 0)


def unit_x2() -> Vector2:
    return Vector2(1, 0)


def unit_y2() -> Vector2:
    return Vector2(0, 1)


def unit_x3() -> Vector3:
    return Vector3(1, 0, 0)


def unit_y3() -> Vector3:
    return Vector3(0, 1, 0)


def unit_z3() -> Vector3:
    return Vector3(0, 0, 1)


# Lights
class LightSource:
    def __init__(
        self,
        pos: Vector3,
        color: str,
        brightness: float,
        falloff_type: str = "linear",
        falloff_rate: float = 0.1,
    ):
        self.pos = pos
        self.color = self.hex_to_rgb(color)
        self.brightness = max(0.0, brightness)
        self.falloff_type = falloff_type
        self.falloff_rate = max(0.001, falloff_rate)

    def hex_to_rgb(self, hex_color: str) -> Tuple[float, float, float]:
        hex_color = hex_color.lstrip("#")
        if len(hex_color) != 6:
            raise ValueError("Hex color must be 6 characters long")
        try:
            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0
            return (r, g, b)
        except ValueError:
            raise ValueError("Invalid hex color format")

    def calculate_falloff(self, distance: float) -> float:
        if self.falloff_type == "none":
            return 1.0
        elif self.falloff_type == "linear":
            return max(0.0, 1.0 - (distance * self.falloff_rate))
        elif self.falloff_type == "quadratic":
            return max(0.0, 1.0 - (distance * distance * self.falloff_rate))
        elif self.falloff_type == "exponential":
            return exp(-distance * self.falloff_rate)
        else:
            return max(0.0, 1.0 - (distance * self.falloff_rate))

    def calculate_lighting(
        self, point: Vector3, normal: Vector3
    ) -> Tuple[float, float, float]:
        return (0.0, 0.0, 0.0)


class PointLight(LightSource):
    def __init__(
        self,
        pos: Vector3,
        color: str,
        brightness: float,
        falloff_type: str = "quadratic",
        falloff_rate: float = 0.1,
    ):
        super().__init__(pos, color, brightness, falloff_type, falloff_rate)

    def calculate_lighting(
        self, point: Vector3, normal: Vector3
    ) -> Tuple[float, float, float]:
        light_dir = self.pos - point
        distance = light_dir.magnitude()

        if distance == 0:
            return (0.0, 0.0, 0.0)

        light_dir = light_dir.normalize()

        falloff = self.calculate_falloff(distance)

        dot_product = max(0.0, normal.dot(light_dir))

        intensity = self.brightness * falloff * dot_product

        return (
            self.color[0] * intensity,
            self.color[1] * intensity,
            self.color[2] * intensity,
        )


class DirectionalLight(LightSource):
    def __init__(
        self,
        pos: Vector3,
        direction: Vector3,
        color: str,
        brightness: float,
        falloff_type: str = "none",
        falloff_rate: float = 0.0,
    ):
        super().__init__(pos, color, brightness, falloff_type, falloff_rate)
        self.direction = direction.normalize()

    def calculate_lighting(
        self, point: Vector3, normal: Vector3
    ) -> Tuple[float, float, float]:
        light_dir = -self.direction

        distance = (point - self.pos).magnitude()
        falloff = self.calculate_falloff(distance)

        dot_product = max(0.0, normal.dot(light_dir))

        intensity = self.brightness * falloff * dot_product

        return (
            self.color[0] * intensity,
            self.color[1] * intensity,
            self.color[2] * intensity,
        )


class SpotLight(LightSource):
    def __init__(
        self,
        pos: Vector3,
        direction: Vector3,
        color: str,
        brightness: float,
        cone_angle: float = radians(30),
        falloff_type: str = "quadratic",
        falloff_rate: float = 0.1,
    ):
        super().__init__(pos, color, brightness, falloff_type, falloff_rate)
        self.direction = direction.normalize()
        self.cone_angle = cone_angle

    def calculate_lighting(
        self, point: Vector3, normal: Vector3
    ) -> Tuple[float, float, float]:
        light_to_point = point - self.pos
        distance = light_to_point.magnitude()

        if distance == 0:
            return (0.0, 0.0, 0.0)

        light_to_point_normalized = light_to_point.normalize()

        angle_to_point = acos(
            max(-1.0, min(1.0, self.direction.dot(light_to_point_normalized)))
        )

        if angle_to_point > self.cone_angle:
            return (0.0, 0.0, 0.0)

        spotlight_factor = cos(angle_to_point / self.cone_angle * (pi / 2))

        light_dir = -light_to_point_normalized

        distance_falloff = self.calculate_falloff(distance)

        dot_product = max(0.0, normal.dot(light_dir))

        intensity = self.brightness * distance_falloff * dot_product * spotlight_factor

        return (
            self.color[0] * intensity,
            self.color[1] * intensity,
            self.color[2] * intensity,
        )


class EmissiveLight(LightSource):
    def __init__(
        self,
        pos: Vector3,
        color: str,
        brightness: float,
        falloff_type: str = "quadratic",
        falloff_rate: float = 0.1,
    ):
        super().__init__(pos, color, brightness, falloff_type, falloff_rate)

    def calculate_lighting(
        self, point: Vector3, normal: Vector3
    ) -> Tuple[float, float, float]:
        light_dir = self.pos - point
        distance = light_dir.magnitude()

        if distance == 0:
            return (0.0, 0.0, 0.0)

        light_dir = light_dir.normalize()

        falloff = self.calculate_falloff(distance)

        dot_product = max(0.0, normal.dot(light_dir))

        intensity = self.brightness * falloff * dot_product

        return (
            self.color[0] * intensity,
            self.color[1] * intensity,
            self.color[2] * intensity,
        )


class LightingConfig:
    def __init__(
        self,
        ambient_color="#333333",
        ambient_intensity=1.0,
    ):
        self.ambient_color = self.hex_to_rgb(ambient_color)
        self.ambient_intensity = max(0.0, ambient_intensity)
        self.light_sources: List[LightSource] = []

    def hex_to_rgb(self, hex_color: str) -> Tuple[float, float, float]:
        hex_color = hex_color.lstrip("#")
        if len(hex_color) != 6:
            raise ValueError("Hex color must be 6 characters long")
        try:
            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0
            return (r, g, b)
        except ValueError:
            raise ValueError("Invalid hex color format")

    def add_light_source(self, light_source: LightSource):
        self.light_sources.append(light_source)

    def remove_light_source(self, light_source: LightSource):
        if light_source in self.light_sources:
            self.light_sources.remove(light_source)

    def clear_light_sources(self):
        self.light_sources.clear()

    def calculate_lighting_at_point(
        self, point: Vector3, normal: Vector3
    ) -> Tuple[float, float, float]:
        total_light = [
            self.ambient_color[0] * self.ambient_intensity,
            self.ambient_color[1] * self.ambient_intensity,
            self.ambient_color[2] * self.ambient_intensity,
        ]

        for light_source in self.light_sources:
            light_contribution = light_source.calculate_lighting(point, normal)
            total_light[0] += light_contribution[0]
            total_light[1] += light_contribution[1]
            total_light[2] += light_contribution[2]

        return (
            max(0.0, min(1.0, total_light[0])),
            max(0.0, min(1.0, total_light[1])),
            max(0.0, min(1.0, total_light[2])),
        )

    def calculate_brightness(self, distance):
        return max(0.1, min(1.0, 10.0 / (distance + 1.0)))


def create_point_source(
    pos: Vector3,
    color: str,
    brightness: float,
    falloff_type: str = "quadratic",
    falloff_rate: float = 0.1,
) -> PointLight:
    return PointLight(pos, color, brightness, falloff_type, falloff_rate)


def create_directional_source(
    pos: Vector3,
    direction: Vector3,
    color: str,
    brightness: float,
    falloff_type: str = "none",
    falloff_rate: float = 0.0,
) -> DirectionalLight:
    return DirectionalLight(
        pos, direction, color, brightness, falloff_type, falloff_rate
    )


def create_spotlight_source(
    pos: Vector3,
    direction: Vector3,
    color: str,
    brightness: float,
    cone_angle: float = radians(30),
    falloff_type: str = "quadratic",
    falloff_rate: float = 0.1,
) -> SpotLight:
    return SpotLight(
        pos, direction, color, brightness, cone_angle, falloff_type, falloff_rate
    )


def set_as_light_source(obj: Dict, brightness: float, color: str = "#ffffff"):
    obj["is_light_source"] = True
    obj["emission_brightness"] = max(0.0, brightness)
    obj["emission_color_hex"] = color

    hex_color = color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError("Hex color must be 6 characters long")

    try:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        obj["emission_color"] = (r, g, b)
    except ValueError:
        raise ValueError("Invalid hex color format")


def update_emissive_lights(objects, lighting_config):
    emissive_lights = [
        light
        for light in lighting_config.light_sources
        if isinstance(light, EmissiveLight)
    ]
    for light in emissive_lights:
        lighting_config.remove_light_source(light)

    for obj in objects:
        if obj.get("is_light_source", False):
            emission_brightness = obj.get("emission_brightness", 0.0)
            emission_color_hex = obj.get("emission_color_hex", "#ffffff")

            for face, material in obj["faces"]:
                face_center = sum(face, Vector3(0, 0, 0)) / len(face)

                emissive_light = EmissiveLight(
                    pos=face_center,
                    color=emission_color_hex,
                    brightness=emission_brightness,
                    falloff_type="quadratic",
                    falloff_rate=0.05,
                )

                lighting_config.add_light_source(emissive_light)


# Camera and control
class Camera:
    def __init__(self, pos, pitch, yaw, lighting_config=None):
        self.pos = pos
        self.pitch = pitch
        self.yaw = yaw
        self.lighting_config = lighting_config or LightingConfig()

        turtle.tracer(0)
        turtle.setup(800, 600)
        pen = turtle.Turtle()
        pen.hideturtle()
        pen.penup()
        pen.color("black")
        pen.speed(0)
        self.pen = pen

    def set_lighting_config(self, lighting_config):
        self.lighting_config = lighting_config

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
        self.pos.y -= pos.y

    def rotate(self, pitch, yaw):
        self.pitch += pitch
        self.yaw += yaw

        max_pitch = radians(89)
        self.pitch = max(-max_pitch, min(max_pitch, self.pitch))

    def get_view_direction(self):
        return Vector3(
            sin(self.yaw) * cos(self.pitch),
            -sin(self.pitch),
            -cos(self.yaw) * cos(self.pitch),
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
        screen_y = screen_y * (screen_height / 2) + screen_height / 2

        turtle_x = screen_x - screen_width / 2
        turtle_y = screen_y - screen_height / 2

        return (turtle_x, turtle_y)

    def compute_face_normal(self, face):
        if len(face) < 3:
            return Vector3(0, 0, 1)

        normal = Vector3(0, 0, 0)

        for i in range(len(face)):
            v1 = face[i]
            v2 = face[(i + 1) % len(face)]

            normal.x += (v1.y - v2.y) * (v1.z + v2.z)
            normal.y += (v1.z - v2.z) * (v1.x + v2.x)
            normal.z += (v1.x - v2.x) * (v1.y + v2.y)

        if normal.magnitude_squared() > 1e-12:
            return normal.normalize()
        else:
            for i in range(len(face) - 2):
                for j in range(i + 1, len(face) - 1):
                    for k in range(j + 1, len(face)):
                        v0 = face[i]
                        v1 = face[j]
                        v2 = face[k]

                        edge1 = v1 - v0
                        edge2 = v2 - v0

                        if (
                            edge1.magnitude_squared() < 1e-12
                            or edge2.magnitude_squared() < 1e-12
                        ):
                            continue

                        normal = edge1.cross(edge2)

                        if normal.magnitude_squared() > 1e-12:
                            return normal.normalize()

            return Vector3(0, 0, 1)

    def is_face_visible(self, face, normal):
        if len(face) < 3:
            return True

        face_center = face[0]
        view_vector = (face_center - self.pos).normalize()
        dot_product = normal.dot(view_vector)

        return dot_product < 0

    def render(
        self,
        objects,
        materials,
        screen_width=800,
        screen_height=600,
        fov=90,
        show_normals=False,
        cull_back_faces=True,
        use_advanced_lighting=True,
    ):
        update_emissive_lights(objects, self.lighting_config)

        render_items = []

        for object in objects:
            is_light_source = object.get("is_light_source", False)
            emission_brightness = object.get("emission_brightness", 0.0)
            emission_color = object.get("emission_color", (1.0, 1.0, 1.0))

            for face, material in object["faces"]:
                normal = self.compute_face_normal(face)

                if cull_back_faces and not self.is_face_visible(face, normal):
                    continue

                projected = [
                    self.project_point(point, screen_width, screen_height, fov)
                    for point in face
                ]

                centroid = sum(face, Vector3(0, 0, 0)) / len(face)
                distance = (centroid - self.pos).magnitude()

                if use_advanced_lighting and self.lighting_config.light_sources:
                    lighting = self.lighting_config.calculate_lighting_at_point(
                        centroid, normal
                    )

                    if material in materials:
                        base_color = materials[material]
                        color = [
                            max(0, min(1, base_color[i] * lighting[i]))
                            for i in range(3)
                        ]
                    else:
                        avg_light = sum(lighting) / 3
                        color = [avg_light, avg_light, avg_light]
                elif use_advanced_lighting and not self.lighting_config.light_sources:
                    lighting = self.lighting_config.calculate_lighting_at_point(
                        centroid, normal
                    )

                    if material in materials:
                        base_color = materials[material]
                        color = [
                            max(0, min(1, base_color[i] * lighting[i]))
                            for i in range(3)
                        ]
                    else:
                        avg_light = sum(lighting) / 3
                        color = [avg_light, avg_light, avg_light]
                else:
                    if material in materials:
                        base_color = materials[material]
                        color = base_color
                    else:
                        color = [1.0, 1.0, 1.0]

                if is_light_source:
                    color = [
                        max(
                            0,
                            min(1, color[i] + emission_color[i] * emission_brightness),
                        )
                        for i in range(3)
                    ]

                render_items.append(
                    {
                        "type": "face",
                        "projected": projected,
                        "color": color,
                        "distance": distance,
                    }
                )

                if show_normals:
                    normal_end = centroid + normal
                    normal_start_proj = self.project_point(
                        centroid, screen_width, screen_height, fov
                    )
                    normal_end_proj = self.project_point(
                        normal_end, screen_width, screen_height, fov
                    )

                    if normal_start_proj and normal_end_proj:
                        render_items.append(
                            {
                                "type": "normal",
                                "start": normal_start_proj,
                                "end": normal_end_proj,
                                "distance": distance,
                            }
                        )

        render_items.sort(key=lambda x: x["distance"], reverse=True)

        for item in render_items:
            if item["type"] == "face":
                projected = item["projected"]
                color = item["color"]

                if None not in projected:
                    self.pen.goto(*projected[0])
                    self.pen.fillcolor(color)
                    self.pen.begin_fill()

                    for pos in projected[1:]:
                        self.pen.goto(*pos)

                    self.pen.end_fill()

            elif item["type"] == "normal":
                self.pen.pencolor("red")
                self.pen.width(2)
                self.pen.penup()
                self.pen.goto(*item["start"])
                self.pen.pendown()
                self.pen.goto(*item["end"])
                self.pen.penup()
                self.pen.width(1)
                self.pen.pencolor("black")


def normalize_framerate(target):
    def decorator(func):
        def wrapped(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            # print(f"fps: {1/(time.time() - start_time)}")
            elapsed = time.time() - start_time
            time_to_sleep = max(0, (1 / (target + 2)) - elapsed)
            time.sleep(time_to_sleep)
            return result

        return wrapped

    return decorator


def mouse_init():
    global mouse_delta_x, mouse_delta_y, last_mouse_x, last_mouse_y, mouse_initialized, mouse_listener
    mouse_delta_x = 0
    mouse_delta_y = 0
    last_mouse_x = 0
    last_mouse_y = 0
    mouse_initialized = False
    if not using_mouse:
        mouse_listener = MouseListener(on_move=on_mouse_move)


def on_mouse_move(x, y):
    global mouse_delta_x, mouse_delta_y, last_mouse_x, last_mouse_y, mouse_initialized, mouse_locked

    if not mouse_locked:
        return

    if not mouse_initialized:
        last_mouse_x = x
        last_mouse_y = y
        mouse_initialized = True
        if not using_mouse:
            mouse_listener.start()
        else:
            mouse.move(400, 300)
        return

    mouse_delta_x = x - last_mouse_x
    mouse_delta_y = y - last_mouse_y

    last_mouse_x = x
    last_mouse_y = y


def handle_movement(speed=0.2, sensitivity=0.05):
    global mouse_delta_x, mouse_delta_y, using_mouse

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
        if using_mouse:
            try:
                x, y = mouse.get_position()
                if mouse_initialized:
                    camera_angle.x -= (y - last_mouse_y) * sensitivity
                    camera_angle.y -= (x - last_mouse_x) * sensitivity
                    mouse.move(400, 300)
                else:
                    globals()["mouse_initialized"] = True
                globals()["last_mouse_x"] = 400
                globals()["last_mouse_y"] = 300
            except Exception:
                camera_angle.x += mouse_delta_y * sensitivity
                camera_angle.y -= mouse_delta_x * sensitivity
                mouse_delta_x = 0
                mouse_delta_y = 0
        else:
            camera_angle.x += mouse_delta_y * sensitivity
            camera_angle.y -= mouse_delta_x * sensitivity
            mouse_delta_x = 0
            mouse_delta_y = 0

    return camera_movement, camera_angle


# Object modification
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
            moved_face = []
            for vertex in face:
                moved_face.append(vertex + axis)
            obj["faces"][face_idx] = (moved_face, material)


# Load files
def load_obj(file_path, scale=0):
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
            vertexes.append(Vector3(*map(lambda x: float(x) * scale, args)))
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


def resource_path(relative_path):
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)


if __name__ == "__main__":
    turtle.bgcolor("#1a1a1a")

    lighting = LightingConfig(
        ambient_color="#1a1a26",
        ambient_intensity=0.3,
    )

    blue_light = create_point_source(
        pos=Vector3(2, 0, 10),
        color="#008cff",
        brightness=0.7,
        falloff_type="linear",
        falloff_rate=0.05,
    )

    red_light = create_point_source(
        pos=Vector3(-2, 0, 10),
        color="#ff0000",
        brightness=0.7,
        falloff_type="linear",
        falloff_rate=0.05,
    )

    sun_light = create_directional_source(
        pos=Vector3(0, 2, -10),
        direction=Vector3(0, -0.5, 1),
        color="#ff78fa",
        brightness=0.8,
        falloff_type="none",
    )

    lighting.add_light_source(red_light)
    lighting.add_light_source(blue_light)
    lighting.add_light_source(sun_light)

    camera = Camera(Vector3(1, 0, -1), radians(0), radians(0), lighting)

    blahaj = load_obj(resource_path("objs/blahaj.obj"), scale=0.5)
    material = load_mtl(resource_path("objs/blahaj.mtl"))

    cube = load_obj(resource_path("objs/cube.obj"), scale=0.2)
    cube_mtl = load_mtl(resource_path("objs/cube.mtl"))

    move_objects(cube, Vector3(5, 3, 0.5))

    set_as_light_source(cube[0], brightness=0.9, color="#ffffff")

    mouse_locked = True
    momentum = zero3()

    mouse_init()

    @normalize_framerate(60)
    def main():
        global mouse_locked, momentum

        camera.pen.clear()

        movement, angle = handle_movement(speed=0.01, sensitivity=0.005)
        momentum += movement
        camera.move_relative(momentum, horizontal_only=True)
        camera.rotate(*angle)
        momentum *= 0.9

        camera.render(
            blahaj + cube,
            {**material, **cube_mtl},
            fov=90,
            use_advanced_lighting=True,
        )

        turtle.update()

        if keyboard.is_pressed("p"):
            mouse_locked = not mouse_locked
            time.sleep(0.2)

        if keyboard.is_pressed("esc"):
            mouse_locked = False

    while True:
        main()
