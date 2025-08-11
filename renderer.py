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

    def rotate_point_around_axis(self, anchor, axis, angle):
        return rotate_point_around_axis(self, anchor, axis, angle)


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


# Math utilities
def triangulate_face(face):
    if len(face) < 3:
        return []
    elif len(face) == 3:
        return [face]
    elif len(face) == 4:
        return [[face[0], face[1], face[2]], [face[0], face[2], face[3]]]
    else:
        triangles = []
        for i in range(1, len(face) - 1):
            triangles.append([face[0], face[i], face[i + 1]])
        return triangles


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


def compute_newell_normal(face):
    normal = Vector3(0, 0, 0)
    for i in range(len(face)):
        v1 = face[i]
        v2 = face[(i + 1) % len(face)]
        normal.x += (v1.y - v2.y) * (v1.z + v2.z)
        normal.y += (v1.z - v2.z) * (v1.x + v2.x)
        normal.z += (v1.x - v2.x) * (v1.y + v2.y)
    return normal


def compute_cross_product_normal(face):
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
                break
            else:
                continue
            break
        else:
            continue
        break
    return Vector3(0, 0, 1)


def calculate_face_normal(face):
    if len(face) < 3:
        return Vector3(0, 0, 1)

    normal = compute_newell_normal(face)

    if normal.magnitude_squared() > 1e-12:
        return normal.normalize()
    else:
        return compute_cross_product_normal(face)


def calculate_face_centroid(face):
    return sum(face, Vector3(0, 0, 0)) / len(face)


# Lights
def ray_intersects_triangle(ray_origin, ray_direction, triangle):
    epsilon = 1e-10

    if len(triangle) < 3:
        return False, 0

    v0, v1, v2 = triangle[0], triangle[1], triangle[2]

    edge1 = v1 - v0
    edge2 = v2 - v0

    if edge1.magnitude_squared() < epsilon or edge2.magnitude_squared() < epsilon:
        return False, 0

    h = ray_direction.cross(edge2)
    a = edge1.dot(h)

    if abs(a) < epsilon:
        return False, 0

    f = 1.0 / a
    s = ray_origin - v0
    u = f * s.dot(h)

    if u < -epsilon or u > 1.0 + epsilon:
        return False, 0

    q = s.cross(edge1)
    v = f * ray_direction.dot(q)

    if v < -epsilon or u + v > 1.0 + epsilon:
        return False, 0

    t = f * edge2.dot(q)

    if t > epsilon:
        return True, t
    else:
        return False, 0


def is_point_occluded(light_pos, target_point, objects, current_object=None):
    light_direction = (target_point - light_pos).normalize()
    light_distance = (target_point - light_pos).magnitude()

    shadow_bias = 0.001

    for obj in objects:
        if current_object is not None and obj is current_object:
            continue

        for face, material in obj.faces:
            if len(face) < 3:
                continue

            face_center = sum(face, Vector3(0, 0, 0)) / len(face)
            face_normal = calculate_face_normal(face)

            light_to_face = (face_center - light_pos).normalize()
            if face_normal.dot(light_to_face) > -0.1:
                continue

            triangles = triangulate_face(face)

            for triangle in triangles:
                intersects, intersection_distance = ray_intersects_triangle(
                    light_pos, light_direction, triangle
                )

                if (
                    intersects
                    and intersection_distance > shadow_bias
                    and intersection_distance < light_distance - shadow_bias
                ):
                    return True

    return False


def clamp_color_component(value):
    return max(0.0, min(1.0, value))


def apply_lighting_to_color(base_color, lighting):
    return [clamp_color_component(base_color[i] * lighting[i]) for i in range(3)]


def apply_emissive_to_color(color, emission_color, emission_brightness):
    return [
        clamp_color_component(color[i] + emission_color[i] * emission_brightness)
        for i in range(3)
    ]


def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
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
        self.color = hex_to_rgb(color)
        self.color_hex = color
        self.brightness = max(0.0, brightness)
        self.falloff_type = falloff_type
        self.falloff_rate = max(0.001, falloff_rate)

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


class AmbientLight(LightSource):
    def __init__(
        self,
        color: str,
        brightness: float,
    ):
        super().__init__(Vector3(0, 0, 0), color, brightness, "none", 0.0)

    def calculate_lighting(
        self, point: Vector3, normal: Vector3
    ) -> Tuple[float, float, float]:
        return (
            self.color[0] * self.brightness,
            self.color[1] * self.brightness,
            self.color[2] * self.brightness,
        )


class PointLight(LightSource):
    def __init__(
        self,
        pos: Vector3,
        color: str,
        brightness: float,
        falloff_type: str = "linear",
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
        falloff_type: str = "linear",
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
        falloff_type: str = "linear",
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

        dot_product_direction = self.direction.dot(light_to_point_normalized)
        dot_product_direction = max(-1.0, min(1.0, dot_product_direction))

        angle_to_point = acos(dot_product_direction)

        if angle_to_point > self.cone_angle:
            return (0.0, 0.0, 0.0)

        spotlight_factor = max(0.0, cos(angle_to_point))

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
        falloff_type: str = "linear",
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
        use_caching=True,
        enable_shadows=False,
        true_darkness=False,
        light_bounces=0,
        precompute_bounces=False,
    ):
        self.light_sources = []
        self.use_caching = use_caching
        self.enable_shadows = enable_shadows
        self.true_darkness = true_darkness
        self.light_bounces = max(0, light_bounces)
        self.precompute_bounces = precompute_bounces
        self._lighting_cache = {}
        self._cache_frame = 0
        self._temp_light = [0.0, 0.0, 0.0]
        self._bounce_cache = {}
        self._bounce_cache_valid = False
        self._face_lighting_cache = {}
        self._centroid_normal_cache = {}

    def add_light_source(self, light_source):
        self.light_sources.append(light_source)
        self._cache_frame = 0
        self._bounce_cache_valid = False
        self._face_lighting_cache.clear()

    def remove_light_source(self, light_source):
        if light_source in self.light_sources:
            self.light_sources.remove(light_source)
            self._cache_frame = 0
            self._bounce_cache_valid = False
            self._face_lighting_cache.clear()

    def clear_light_sources(self):
        self.light_sources.clear()
        self._cache_frame = 0
        self._bounce_cache_valid = False
        self._face_lighting_cache.clear()

    def set_caching(self, enabled):
        self.use_caching = enabled
        if not enabled:
            self._lighting_cache.clear()
            self._face_lighting_cache.clear()
            self._centroid_normal_cache.clear()

    def set_shadows(self, enabled):
        self.enable_shadows = enabled
        self._lighting_cache.clear()
        self._face_lighting_cache.clear()

    def set_true_darkness(self, enabled):
        self.true_darkness = enabled
        self._lighting_cache.clear()
        self._face_lighting_cache.clear()

    def set_light_bounces(self, bounces):
        self.light_bounces = max(0, bounces)
        self._lighting_cache.clear()
        self._bounce_cache_valid = False
        self._face_lighting_cache.clear()

    def set_precompute_bounces(self, enabled):
        self.precompute_bounces = enabled
        if enabled:
            self._bounce_cache_valid = False
        else:
            self._bounce_cache.clear()

    def _get_cached_centroid_normal(self, obj, face_idx):
        cache_key = (id(obj), face_idx)
        if cache_key in self._centroid_normal_cache:
            return self._centroid_normal_cache[cache_key]

        centroid, normal = obj.get_face_data(face_idx)
        self._centroid_normal_cache[cache_key] = (centroid, normal)
        return centroid, normal

    def _get_cached_face_lighting(self, obj, face_idx, centroid, normal):
        cache_key = (
            id(obj),
            face_idx,
            round(centroid.x, 1),
            round(centroid.y, 1),
            round(centroid.z, 1),
            round(normal.x, 2),
            round(normal.y, 2),
            round(normal.z, 2),
        )

        if cache_key in self._face_lighting_cache:
            return self._face_lighting_cache[cache_key]

        face_lighting = [0.0, 0.0, 0.0]
        for light_source in self.light_sources:
            if isinstance(light_source, AmbientLight):
                ambient_contrib = light_source.calculate_lighting(centroid, normal)
                face_lighting[0] += ambient_contrib[0]
                face_lighting[1] += ambient_contrib[1]
                face_lighting[2] += ambient_contrib[2]
            else:
                light_distance = (light_source.pos - centroid).magnitude()
                if light_distance <= 50.0:
                    light_contribution = light_source.calculate_lighting(
                        centroid, normal
                    )
                    face_lighting[0] += light_contribution[0]
                    face_lighting[1] += light_contribution[1]
                    face_lighting[2] += light_contribution[2]

        self._face_lighting_cache[cache_key] = face_lighting
        return face_lighting

    def precompute_bounce_lighting(self, objects):
        if not self.precompute_bounces or self.light_bounces == 0:
            return

        self._bounce_cache.clear()

        if isinstance(objects, Object):
            objects = [objects]

        flat_objects = []
        for obj_list in objects:
            if isinstance(obj_list, list):
                flat_objects.extend(obj_list)
            else:
                flat_objects.append(obj_list)
        objects = flat_objects

        for obj in objects:
            for face_idx in range(len(obj.faces)):
                centroid, normal = self._get_cached_centroid_normal(obj, face_idx)

                bounce_strength = 0.3
                total_bounce_lighting = [0.0, 0.0, 0.0]

                for bounce in range(self.light_bounces):
                    bounce_strength *= 0.7

                    bounce_lighting = self._calculate_bounce_lighting_optimized(
                        centroid,
                        normal,
                        objects,
                        obj,
                        face_idx,
                        max_distance=8.0,
                        bounce_strength=bounce_strength,
                    )

                    total_bounce_lighting[0] += bounce_lighting[0]
                    total_bounce_lighting[1] += bounce_lighting[1]
                    total_bounce_lighting[2] += bounce_lighting[2]

                cache_key = (
                    round(centroid.x, 0),
                    round(centroid.y, 0),
                    round(centroid.z, 0),
                    round(normal.x, 2),
                    round(normal.y, 2),
                    round(normal.z, 2),
                )

                self._bounce_cache[cache_key] = total_bounce_lighting

        self._bounce_cache_valid = True

    def _calculate_bounce_lighting_optimized(
        self,
        point,
        normal,
        objects,
        current_object,
        current_face_idx,
        max_distance=10.0,
        bounce_strength=0.5,
    ):
        bounce_lighting = [0.0, 0.0, 0.0]

        num_samples = 6
        sample_directions = []

        up = Vector3(0, 1, 0)
        if abs(normal.dot(up)) > 0.9:
            up = Vector3(1, 0, 0)

        tangent = normal.cross(up).normalize()
        bitangent = normal.cross(tangent).normalize()

        for i in range(num_samples):
            angle = (i / num_samples) * 2 * pi
            offset_amount = 0.3
            offset_dir = (
                tangent * cos(angle) * offset_amount
                + bitangent * sin(angle) * offset_amount
            )
            sample_dir = (normal + offset_dir).normalize()
            sample_directions.append(sample_dir)

        for sample_dir in sample_directions:
            for obj in objects:
                if obj is current_object:
                    continue

                for face_idx in range(len(obj.faces)):
                    face_center, face_normal = self._get_cached_centroid_normal(
                        obj, face_idx
                    )

                    distance = (face_center - point).magnitude()
                    if distance > max_distance or distance < 0.1:
                        continue

                    direction_to_face = (face_center - point).normalize()

                    if normal.dot(direction_to_face) < 0.0:
                        continue
                    if face_normal.dot(-direction_to_face) < 0.0:
                        continue

                    angle_diff = abs(
                        acos(max(-1, min(1, sample_dir.dot(direction_to_face))))
                    )
                    if angle_diff > pi / 3:
                        continue

                    face_lighting = self._get_cached_face_lighting(
                        obj, face_idx, face_center, face_normal
                    )

                    if face_lighting[0] + face_lighting[1] + face_lighting[2] < 0.01:
                        continue

                    falloff = max(0.0, 1.0 - (distance / max_distance))
                    angle_factor = max(0.0, normal.dot(direction_to_face))
                    surface_angle_factor = max(0.0, face_normal.dot(-direction_to_face))

                    bounce_factor = (
                        bounce_strength
                        * falloff
                        * angle_factor
                        * surface_angle_factor
                        * 0.3
                    )

                    bounce_lighting[0] += face_lighting[0] * bounce_factor
                    bounce_lighting[1] += face_lighting[1] * bounce_factor
                    bounce_lighting[2] += face_lighting[2] * bounce_factor

        sample_weight = 1.0 / num_samples
        bounce_lighting[0] *= sample_weight
        bounce_lighting[1] *= sample_weight
        bounce_lighting[2] *= sample_weight

        return (
            max(0.0, min(1.0, bounce_lighting[0])),
            max(0.0, min(1.0, bounce_lighting[1])),
            max(0.0, min(1.0, bounce_lighting[2])),
        )

    def get_precomputed_bounce_lighting(self, point, normal):
        if not self.precompute_bounces or not self._bounce_cache_valid:
            return [0.0, 0.0, 0.0]

        cache_key = (
            round(point.x, 0),
            round(point.y, 0),
            round(point.z, 0),
            round(normal.x, 2),
            round(normal.y, 2),
            round(normal.z, 2),
        )

        return self._bounce_cache.get(cache_key, [0.0, 0.0, 0.0])

    def calculate_lighting_at_point(
        self, point, normal, objects=None, current_object=None, current_face_idx=None
    ):
        if not self.use_caching:
            return self._compute_lighting(
                point, normal, objects, current_object, current_face_idx
            )

        cache_key = (
            round(point.x, 0),
            round(point.y, 0),
            round(point.z, 0),
            round(normal.x, 2),
            round(normal.y, 2),
            round(normal.z, 2),
        )

        cached_result = self._lighting_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        result = self._compute_lighting(
            point, normal, objects, current_object, current_face_idx
        )
        self._lighting_cache[cache_key] = result
        return result

    def _compute_lighting(
        self, point, normal, objects=None, current_object=None, current_face_idx=None
    ):
        self._temp_light[0] = 0.0
        self._temp_light[1] = 0.0
        self._temp_light[2] = 0.0

        ambient_light_total = [0.0, 0.0, 0.0]

        for light_source in self.light_sources:
            if isinstance(light_source, AmbientLight):
                light_contribution = light_source.calculate_lighting(point, normal)
                ambient_light_total[0] += light_contribution[0]
                ambient_light_total[1] += light_contribution[1]
                ambient_light_total[2] += light_contribution[2]
            else:
                distance = (light_source.pos - point).magnitude()
                if distance > 50.0:
                    continue

                if (
                    self.enable_shadows
                    and objects
                    and is_point_occluded(
                        light_source.pos,
                        point,
                        objects,
                        current_object,
                    )
                ):
                    continue

                light_contribution = light_source.calculate_lighting(point, normal)
                if (
                    light_contribution[0]
                    + light_contribution[1]
                    + light_contribution[2]
                    > 0.001
                ):
                    self._temp_light[0] += light_contribution[0]
                    self._temp_light[1] += light_contribution[1]
                    self._temp_light[2] += light_contribution[2]

        if self.light_bounces > 0 and objects:
            if self.precompute_bounces:
                bounce_lighting = self.get_precomputed_bounce_lighting(point, normal)
                self._temp_light[0] += bounce_lighting[0]
                self._temp_light[1] += bounce_lighting[1]
                self._temp_light[2] += bounce_lighting[2]
            else:
                bounce_strength = 0.3
                for bounce in range(self.light_bounces):
                    bounce_strength *= 0.7

                    bounce_lighting = calculate_bounce_lighting(
                        point,
                        normal,
                        objects,
                        current_object,
                        current_face_idx,
                        self.light_sources,
                        max_distance=8.0,
                        bounce_strength=bounce_strength,
                    )

                    self._temp_light[0] += bounce_lighting[0]
                    self._temp_light[1] += bounce_lighting[1]
                    self._temp_light[2] += bounce_lighting[2]

        if (
            not self.true_darkness
            and (
                ambient_light_total[0] + ambient_light_total[1] + ambient_light_total[2]
            )
            == 0
        ):
            ambient_light_total = [1.0, 1.0, 1.0]

        self._temp_light[0] += ambient_light_total[0]
        self._temp_light[1] += ambient_light_total[1]
        self._temp_light[2] += ambient_light_total[2]

        return (
            max(0.0, min(1.0, self._temp_light[0])),
            max(0.0, min(1.0, self._temp_light[1])),
            max(0.0, min(1.0, self._temp_light[2])),
        )

    def calculate_brightness(self, distance):
        return max(0.1, min(1.0, 10.0 / (distance + 1.0)))


def set_as_light_source(obj: Dict, brightness: float, color: str = "#ffffff"):
    obj["is_light_source"] = True
    obj["emission_brightness"] = max(0.0, brightness)
    obj["emission_color_hex"] = color

    r, g, b = hex_to_rgb(color)
    obj["emission_color"] = (r, g, b)


def update_emissive_lights(objects, lighting_config):
    if isinstance(objects, Object):
        objects = [objects]

    emissive_lights = [
        light
        for light in lighting_config.light_sources
        if isinstance(light, EmissiveLight)
    ]
    for light in emissive_lights:
        lighting_config.remove_light_source(light)

    emissive_light_pool = []

    for obj in objects:
        if obj.is_light_source:
            emission_brightness = obj.emission_brightness
            emission_color_hex = obj.emission_color_hex

            for face, material in obj.faces:
                if len(emissive_light_pool) > 0:
                    emissive_light = emissive_light_pool.pop()
                    emissive_light.pos = sum(face, Vector3(0, 0, 0)) / len(face)
                    emissive_light.color = hex_to_rgb(emission_color_hex)
                    emissive_light.color_hex = emission_color_hex
                    emissive_light.brightness = emission_brightness
                else:
                    face_center = sum(face, Vector3(0, 0, 0)) / len(face)
                    emissive_light = EmissiveLight(
                        pos=face_center,
                        color=emission_color_hex,
                        brightness=emission_brightness,
                        falloff_type="linear",
                        falloff_rate=0.05,
                    )

                lighting_config.add_light_source(emissive_light)


def calculate_bounce_lighting(
    point,
    normal,
    objects,
    current_object,
    current_face_idx,
    light_sources,
    max_distance=10.0,
    bounce_strength=0.5,
):
    bounce_lighting = [0.0, 0.0, 0.0]

    num_samples = 8
    sample_directions = []

    up = Vector3(0, 1, 0)
    if abs(normal.dot(up)) > 0.9:
        up = Vector3(1, 0, 0)

    tangent = normal.cross(up).normalize()
    bitangent = normal.cross(tangent).normalize()

    for i in range(num_samples):
        angle = (i / num_samples) * 2 * pi
        offset_amount = 0.3
        offset_dir = (
            tangent * cos(angle) * offset_amount
            + bitangent * sin(angle) * offset_amount
        )
        sample_dir = (normal + offset_dir).normalize()
        sample_directions.append(sample_dir)

    for sample_dir in sample_directions:
        for obj in objects:
            if obj is current_object:
                continue

            for face_idx, (face, material) in enumerate(obj.faces):
                if len(face) < 3:
                    continue

                face_center = sum(face, Vector3(0, 0, 0)) / len(face)
                face_normal = calculate_face_normal(face)

                distance = (face_center - point).magnitude()
                if distance > max_distance or distance < 0.1:
                    continue

                direction_to_face = (face_center - point).normalize()

                if normal.dot(direction_to_face) < 0.0:
                    continue
                if face_normal.dot(-direction_to_face) < 0.0:
                    continue

                angle_diff = abs(
                    acos(max(-1, min(1, sample_dir.dot(direction_to_face))))
                )
                if angle_diff > pi / 3:
                    continue

                face_lighting = [0.0, 0.0, 0.0]
                for light_source in light_sources:
                    if isinstance(light_source, AmbientLight):
                        ambient_contrib = light_source.calculate_lighting(
                            face_center, face_normal
                        )
                        face_lighting[0] += ambient_contrib[0]
                        face_lighting[1] += ambient_contrib[1]
                        face_lighting[2] += ambient_contrib[2]
                    else:
                        light_contribution = light_source.calculate_lighting(
                            face_center, face_normal
                        )
                        face_lighting[0] += light_contribution[0]
                        face_lighting[1] += light_contribution[1]
                        face_lighting[2] += light_contribution[2]

                if face_lighting[0] + face_lighting[1] + face_lighting[2] < 0.01:
                    continue

                falloff = max(0.0, 1.0 - (distance / max_distance))
                angle_factor = max(0.0, normal.dot(direction_to_face))
                surface_angle_factor = max(0.0, face_normal.dot(-direction_to_face))

                bounce_factor = (
                    bounce_strength
                    * falloff
                    * angle_factor
                    * surface_angle_factor
                    * 0.3
                )

                bounce_lighting[0] += face_lighting[0] * bounce_factor
                bounce_lighting[1] += face_lighting[1] * bounce_factor
                bounce_lighting[2] += face_lighting[2] * bounce_factor

    sample_weight = 1.0 / num_samples
    bounce_lighting[0] *= sample_weight
    bounce_lighting[1] *= sample_weight
    bounce_lighting[2] *= sample_weight

    return (
        max(0.0, min(1.0, bounce_lighting[0])),
        max(0.0, min(1.0, bounce_lighting[1])),
        max(0.0, min(1.0, bounce_lighting[2])),
    )


# Camera and control
class Camera:
    def __init__(self, pos, direction=None, lighting_config=None, use_caching=True):
        self.pos = pos
        if direction is None:
            self.direction = Vector3(0, 0, 1)
        else:
            self.direction = direction.normalize()
        self.lighting_config = lighting_config or LightingConfig()
        self.use_caching = use_caching

        turtle.tracer(0)
        turtle.setup(800, 600)
        pen = turtle.Turtle()
        pen.hideturtle()
        pen.penup()
        pen.color("black")
        pen.speed(0)
        self.pen = pen

        self._render_items = []
        self._temp_projected = []

    def set_lighting_config(self, lighting_config):
        self.lighting_config = lighting_config

    def set_caching(self, enabled):
        self.use_caching = enabled

    def move_axis(self, pos):
        self.pos += pos
        return self.pos

    def move(self, steps, horizontal_only=False):
        if horizontal_only:
            forward = Vector3(self.direction.x, 0, self.direction.z).normalize()
        else:
            forward = self.direction

        move = forward * steps
        self.pos += move
        return self.pos

    def strafe(self, steps):
        up = Vector3(0, 1, 0)
        right = self.direction.cross(up).normalize()

        move = right * steps
        self.pos += move
        return self.pos

    def move_relative(self, pos, horizontal_only=False):
        self.move(pos.z, horizontal_only)
        self.strafe(pos.x)
        self.pos.y -= pos.y

    def rotate(self, pitch_delta, yaw_delta):
        up = Vector3(0, 1, 0)

        if yaw_delta != 0:
            cos_yaw = cos(yaw_delta)
            sin_yaw = sin(yaw_delta)
            new_x = self.direction.x * cos_yaw - self.direction.z * sin_yaw
            new_z = self.direction.x * sin_yaw + self.direction.z * cos_yaw
            self.direction = Vector3(new_x, self.direction.y, new_z).normalize()

        if pitch_delta != 0:
            right = self.direction.cross(up).normalize()
            cos_pitch = cos(pitch_delta)
            sin_pitch = sin(pitch_delta)
            new_direction = self.direction * cos_pitch + up * sin_pitch
            self.direction = new_direction.normalize()

    def get_view_direction(self):
        return self.direction

    def project_point(self, point, screen_width=800, screen_height=600, fov=90):
        relative = point - self.pos

        forward = self.direction
        up = Vector3(0, 1, 0)
        right = forward.cross(up).normalize()
        actual_up = right.cross(forward).normalize()

        x = relative.dot(right)
        y = relative.dot(actual_up)
        z = relative.dot(forward)

        if z <= 0:
            return None

        fov_rad = radians(fov)
        f = 1 / tan(fov_rad / 2)

        screen_x = (x * f / z) * (screen_height / screen_width)
        screen_y = y * f / z

        screen_x = screen_x * (screen_width / 2) + screen_width / 2
        screen_y = screen_y * (screen_height / 2) + screen_height / 2

        turtle_x = screen_x - screen_width / 2
        turtle_y = screen_y - screen_height / 2

        return (turtle_x, turtle_y)

    def compute_face_normal(self, face):
        return calculate_face_normal(face)

    def is_face_visible(self, face_center, normal):
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
        draw_light_sources=False,
    ):
        if isinstance(objects, Object):
            objects = [objects]
        if isinstance(materials, Material):
            materials = {materials.name: materials.color}
        elif isinstance(materials, list):
            material_dict = {}
            for mat_list in materials:
                if isinstance(mat_list, list):
                    for mat in mat_list:
                        material_dict[mat.name] = mat.color
                else:
                    material_dict[mat_list.name] = mat_list.color
            materials = material_dict

        flat_objects = []
        for obj_list in objects:
            if isinstance(obj_list, list):
                flat_objects.extend(obj_list)
            else:
                flat_objects.append(obj_list)
        objects = flat_objects

        update_emissive_lights(objects, self.lighting_config)

        if (
            self.lighting_config.precompute_bounces
            and not self.lighting_config._bounce_cache_valid
            and self.lighting_config.light_bounces > 0
        ):
            self.lighting_config.precompute_bounce_lighting(objects)

        self._render_items.clear()
        visible_faces = 0

        for object in objects:
            is_light_source = object.is_light_source
            emission_brightness = object.emission_brightness
            emission_color = object.emission_color

            for face_idx, (face, material) in enumerate(object.faces):
                if material:
                    material = f"{object.name}.{material}"

                centroid, normal = object.get_face_data(face_idx)

                distance = (centroid - self.pos).magnitude()
                if distance > 50.0:
                    continue

                if cull_back_faces and not self.is_face_visible(centroid, normal):
                    continue

                visible_faces += 1

                self._temp_projected.clear()
                all_visible = True
                for point in face:
                    proj = self.project_point(point, screen_width, screen_height, fov)
                    if proj is None:
                        all_visible = False
                        break
                    self._temp_projected.append(proj)

                if not all_visible:
                    continue

                if use_advanced_lighting:
                    lighting = self.lighting_config.calculate_lighting_at_point(
                        centroid, normal, objects, object, face_idx
                    )

                    if material in materials:
                        base_color = materials[material]
                        color = apply_lighting_to_color(base_color, lighting)
                    else:
                        avg_light = (lighting[0] + lighting[1] + lighting[2]) / 3
                        color = [avg_light, avg_light, avg_light]
                else:
                    if material in materials:
                        base_color = materials[material]
                        color = base_color
                    else:
                        color = [1.0, 1.0, 1.0]

                if is_light_source:
                    color = apply_emissive_to_color(
                        color, emission_color, emission_brightness
                    )

                self._render_items.append(
                    {
                        "type": "face",
                        "projected": list(self._temp_projected),
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
                        self._render_items.append(
                            {
                                "type": "normal",
                                "start": normal_start_proj,
                                "end": normal_end_proj,
                                "distance": distance,
                            }
                        )

        if draw_light_sources:
            for light_source in self.lighting_config.light_sources:
                if isinstance(light_source, AmbientLight):
                    continue

                light_proj = self.project_point(
                    light_source.pos, screen_width, screen_height, fov
                )
                if light_proj:
                    distance = (light_source.pos - self.pos).magnitude()
                    self._render_items.append(
                        {
                            "type": "light_source",
                            "position": light_proj,
                            "color": [
                                color * light_source.brightness
                                for color in light_source.color
                            ],
                            "distance": distance,
                        }
                    )

        self._render_items.sort(key=lambda x: x["distance"], reverse=True)

        for item in self._render_items:
            if item["type"] == "face":
                projected = item["projected"]
                color = item["color"]

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

            elif item["type"] == "light_source":
                self.pen.penup()
                self.pen.goto(*item["position"])
                self.pen.pencolor(item["color"])
                self.pen.dot(6)
                self.pen.pencolor("black")


def accurate_sleep(seconds):
    if seconds == 0:
        return
    elif seconds < 0.05:
        target = time.perf_counter() + seconds
        while time.perf_counter() < target:
            pass
    else:
        time.sleep(seconds)


def normalize_framerate(target):
    def decorator(func):
        def wrapped(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            print(f"fps: {1/(time.time() - start_time)}")
            elapsed_after_func = time.time() - start_time
            time_to_sleep = max(0, (1 / target) - elapsed_after_func)
            accurate_sleep(time_to_sleep)
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
            x, y = mouse.get_position()
            if mouse_initialized:
                camera_angle.x -= (y - 300) * sensitivity
                camera_angle.y += (x - 400) * sensitivity
                mouse.move(400, 300)
            else:
                globals()["mouse_initialized"] = True
            globals()["last_mouse_x"] = 400
            globals()["last_mouse_y"] = 300
        else:
            camera_angle.x += mouse_delta_y * sensitivity
            camera_angle.y -= mouse_delta_x * sensitivity
            mouse_delta_x = 0
            mouse_delta_y = 0

    return camera_movement, camera_angle


# Load files n stuff
class Material:
    def __init__(
        self,
        name: str,
        color: List[float],
        specular: List[float] = None,
        shininess: float = 0.0,
    ):
        self.name = name
        self.color = color
        self.specular = specular or [0.0, 0.0, 0.0]
        self.shininess = max(0.0, min(1000.0, shininess))


class Object:
    def __init__(self, name: str, faces: List = None, use_caching=True):
        self.name = name
        self.faces = faces or []
        self.is_light_source = False
        self.emission_brightness = 0.0
        self.emission_color_hex = "#ffffff"
        self.emission_color = (1.0, 1.0, 1.0)
        self.use_caching = use_caching
        self._face_cache = {}
        self._cache_dirty = False

    def set_caching(self, enabled: bool):
        self.use_caching = enabled
        if not enabled:
            self._face_cache = {}

    def invalidate_cache(self):
        if self.use_caching:
            self._cache_dirty = True

    def set_as_light_source(self, brightness: float, color: str = "#ffffff"):
        self.is_light_source = True
        self.emission_brightness = max(0.0, brightness)
        self.emission_color_hex = color
        self.emission_color = hex_to_rgb(color)

    def rotate(self, anchor: Vector3, rotation_vector: Vector3):
        angle = sqrt(rotation_vector.x**2 + rotation_vector.y**2 + rotation_vector.z**2)

        if angle == 0:
            return

        axis = Vector3(
            rotation_vector.x / angle,
            rotation_vector.y / angle,
            rotation_vector.z / angle,
        )

        for face_idx, (face, material) in enumerate(self.faces):
            rotated_face = []
            for vertex in face:
                rotated_vertex = vertex.rotate_point_around_axis(anchor, axis, angle)
                rotated_face.append(rotated_vertex)
            self.faces[face_idx] = (rotated_face, material)

        self.invalidate_cache()

    def move(self, axis: Vector3):
        for face_idx, (face, material) in enumerate(self.faces):
            moved_face = []
            for vertex in face:
                moved_face.append(vertex + axis)
            self.faces[face_idx] = (moved_face, material)

        self.invalidate_cache()

    def get_face_data(self, face_idx):
        if not self.use_caching:
            return self._compute_face_data(face_idx)

        if self._cache_dirty:
            self._face_cache = {}
            self._cache_dirty = False

        if face_idx in self._face_cache:
            return self._face_cache[face_idx]

        result = self._compute_face_data(face_idx)
        self._face_cache[face_idx] = result
        return result

    def _compute_face_data(self, face_idx):
        face, _ = self.faces[face_idx]
        centroid = calculate_face_centroid(face)
        normal = calculate_face_normal(face)
        return (centroid, normal)


def load_obj(file_path, scale=0) -> List[Object]:
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
            objects.append(Object(name, []))
        elif command == "v":
            vertexes.append(Vector3(*map(lambda x: float(x) * scale, args)))
        elif command == "f":
            objects[-1].faces.append(
                ([vertexes[int(i.split("/")[0]) - 1] for i in args], curr_mtl)
            )
        elif command == "usemtl":
            curr_mtl = args[0]

    return objects


def load_mtl(file_path, objects) -> List[Material]:
    if isinstance(objects, Object):
        objects = [objects]

    with open(file_path) as file:
        lines = file.readlines()

    materials = []
    curr_mtl = None
    curr_kd = [1.0, 1.0, 1.0]
    curr_ks = [0.0, 0.0, 0.0]
    curr_ns = 0.0

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        command, *args = parts
        if command == "newmtl":
            if curr_mtl:
                for obj in objects:
                    materials.append(
                        Material(f"{obj.name}.{curr_mtl}", curr_kd, curr_ks, curr_ns)
                    )
            curr_mtl = args[0]
            curr_kd = [1.0, 1.0, 1.0]
            curr_ks = [0.0, 0.0, 0.0]
            curr_ns = 0.0
        elif command == "Kd" and curr_mtl:
            curr_kd = [float(i) for i in args]
        elif command == "Ks" and curr_mtl:
            curr_ks = [float(i) for i in args]
        elif command == "Ns" and curr_mtl:
            curr_ns = float(args[0])

    if curr_mtl:
        for obj in objects:
            materials.append(
                Material(f"{obj.name}.{curr_mtl}", curr_kd, curr_ks, curr_ns)
            )

    return materials


def resource_path(relative_path):
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)


# main
if __name__ == "__main__":
    import gc

    gc.set_threshold(2000, 20, 20)

    turtle.bgcolor("#181818")
    use_caching = True

    lighting = LightingConfig(
        use_caching=use_caching,
        enable_shadows=True,
        true_darkness=False,
        light_bounces=2,
        precompute_bounces=True,
    )

    lighting.add_light_source(
        AmbientLight(
            color="#333333",
            brightness=0.3,
        )
    )

    lighting.add_light_source(
        PointLight(
            pos=Vector3(-0.5, 1, -1.5),
            color="#008cff",
            brightness=0.7,
            falloff_type="linear",
            falloff_rate=0.05,
        )
    )
    lighting.add_light_source(
        PointLight(
            pos=Vector3(0.5, 1, -1.5),
            color="#04ff00",
            brightness=0.7,
            falloff_type="linear",
            falloff_rate=0.05,
        )
    )

    lighting.add_light_source(
        PointLight(
            pos=Vector3(-0.5, -1, -1.5),
            color="#ffb300",
            brightness=0.7,
            falloff_type="linear",
            falloff_rate=0.05,
        )
    )
    lighting.add_light_source(
        PointLight(
            pos=Vector3(0.5, -1, -1.5),
            color="#f2ff00",
            brightness=0.7,
            falloff_type="linear",
            falloff_rate=0.05,
        )
    )
    lighting.add_light_source(
        PointLight(
            pos=Vector3(0.5, 0, 2),
            color="#ff0000",
            brightness=0.7,
            falloff_type="linear",
            falloff_rate=0.05,
        )
    )
    lighting.add_light_source(
        PointLight(
            pos=Vector3(-0.5, 0, 2),
            color="#aa00ff",
            brightness=0.7,
            falloff_type="linear",
            falloff_rate=0.05,
        )
    )
    camera = Camera(
        Vector3(0.9, 0, 0), Vector3(0, 0, 1), lighting, use_caching=use_caching
    )

    cube = load_obj(resource_path("objs/cube.obj"), scale=0.2)
    cube_mtl = load_mtl(resource_path("objs/cube.mtl"), cube)
    cube[0].move(Vector3(0, 0, -4))

    blahaj = load_obj(resource_path("objs/blahaj.obj"), scale=0.5)
    blahaj_mtl = load_mtl(resource_path("objs/blahaj.mtl"), blahaj)

    mouse_locked = True
    momentum = zero3()
    frame_count = 0

    mouse_init()

    @normalize_framerate(60)
    def main():
        global mouse_locked, momentum, frame_count

        camera.pen.clear()

        movement, angle = handle_movement(speed=0.01, sensitivity=0.005)
        momentum += movement
        camera.move_relative(momentum, horizontal_only=True)
        camera.rotate(*angle)
        momentum *= 0.9

        camera.render(
            cube + blahaj,
            cube_mtl + blahaj_mtl,
            fov=103,
            use_advanced_lighting=True,
            draw_light_sources=True,
        )

        turtle.update()

        frame_count += 1
        if frame_count % 300 == 0:
            gc.collect()

        if keyboard.is_pressed("p"):
            mouse_locked = not mouse_locked
            time.sleep(0.2)

        if keyboard.is_pressed("esc"):
            mouse_locked = False

    while True:
        main()
