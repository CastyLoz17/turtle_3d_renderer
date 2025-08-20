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
    """A 2D vector class with basic mathematical operations."""

    __slots__ = ["x", "y"]

    def __init__(self, x: Number, y: Number):
        """Initialize a 2D vector.

        Args:
            x: X component
            y: Y component
        """
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

    def __neg__(self) -> "Vector2":
        return Vector2(-self.x, -self.y)

    def __eq__(self, other: "Vector2") -> bool:
        epsilon = 1e-10
        return abs(self.x - other.x) < epsilon and abs(self.y - other.y) < epsilon

    def __hash__(self):
        return hash((round(self.x, 10), round(self.y, 10)))

    def dot(self, other: "Vector2") -> float:
        """Calculate dot product with another vector."""
        return self.x * other.x + self.y * other.y

    def magnitude(self) -> float:
        """Calculate the magnitude of the vector."""
        return sqrt(self.x * self.x + self.y * self.y)

    def magnitude_squared(self) -> float:
        """Calculate the squared magnitude (faster than magnitude)."""
        return self.x * self.x + self.y * self.y

    def normalize(self) -> "Vector2":
        """Return a normalized version of this vector."""
        mag_sq = self.x * self.x + self.y * self.y
        if mag_sq == 0:
            return Vector2(0, 0)
        inv_mag = 1.0 / sqrt(mag_sq)
        return Vector2(self.x * inv_mag, self.y * inv_mag)

    def distance_to(self, other: "Vector2") -> float:
        """Calculate distance to another vector."""
        dx = self.x - other.x
        dy = self.y - other.y
        return sqrt(dx * dx + dy * dy)

    def angle(self) -> float:
        """Get the angle of this vector in radians."""
        return atan2(self.y, self.x)

    def rotate(self, angle: float) -> "Vector2":
        """Rotate the vector by the given angle in radians."""
        cos_a, sin_a = cos(angle), sin(angle)
        return Vector2(self.x * cos_a - self.y * sin_a, self.x * sin_a + self.y * cos_a)

    def is_zero(self) -> bool:
        """Check if the vector is zero (within epsilon)."""
        epsilon = 1e-10
        return abs(self.x) < epsilon and abs(self.y) < epsilon


class Vector3:
    """A 3D vector class with basic mathematical operations."""

    __slots__ = ["x", "y", "z"]

    def __init__(self, x: Number, y: Number, z: Number):
        """Initialize a 3D vector.

        Args:
            x: X component
            y: Y component
            z: Z component
        """
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
        """Calculate dot product with another vector."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3") -> "Vector3":
        """Calculate cross product with another vector."""
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def magnitude(self) -> float:
        """Calculate the magnitude of the vector."""
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def magnitude_squared(self) -> float:
        """Calculate the squared magnitude (faster than magnitude)."""
        return self.x * self.x + self.y * self.y + self.z * self.z

    def normalize(self) -> "Vector3":
        """Return a normalized version of this vector."""
        mag_sq = self.x * self.x + self.y * self.y + self.z * self.z
        if mag_sq == 0:
            return Vector3(0, 0, 0)
        inv_mag = 1.0 / sqrt(mag_sq)
        return Vector3(self.x * inv_mag, self.y * inv_mag, self.z * inv_mag)

    def distance_to(self, other: "Vector3") -> float:
        """Calculate distance to another vector."""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return sqrt(dx * dx + dy * dy + dz * dz)

    def is_zero(self) -> bool:
        """Check if the vector is zero (within epsilon)."""
        epsilon = 1e-10
        return abs(self.x) < epsilon and abs(self.y) < epsilon and abs(self.z) < epsilon

    def project_onto(self, other: "Vector3") -> "Vector3":
        """Project this vector onto another vector."""
        if other.is_zero():
            return Vector3(0, 0, 0)
        return other * (self.dot(other) / other.magnitude_squared())

    def reflect(self, normal: "Vector3") -> "Vector3":
        """Reflect this vector off a surface with the given normal."""
        return self - 2 * self.project_onto(normal)

    def rotate_point_around_axis(self, anchor, axis, angle):
        """Rotate this point around an arbitrary axis."""
        return rotate_point_around_axis(self, anchor, axis, angle)


def zero2() -> Vector2:
    """Create a zero Vector2."""
    return Vector2(0, 0)


def zero3() -> Vector3:
    """Create a zero Vector3."""
    return Vector3(0, 0, 0)


def unit_x3() -> Vector3:
    """Create a unit vector in the X direction."""
    return Vector3(1, 0, 0)


def unit_y3() -> Vector3:
    """Create a unit vector in the Y direction."""
    return Vector3(0, 1, 0)


def unit_z3() -> Vector3:
    """Create a unit vector in the Z direction."""
    return Vector3(0, 0, 1)


# Math utilities
def triangulate_face(face):
    """Convert a face with n vertices into triangles.

    Args:
        face: List of vertices representing a face

    Returns:
        List of triangles (each triangle is a list of 3 vertices)
    """
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
    """Rotate a point around an arbitrary axis using Rodrigues' rotation formula.

    Args:
        point: The point to rotate
        anchor: The anchor point of the rotation axis
        axis: The axis of rotation (will be normalized)
        angle: Rotation angle in radians

    Returns:
        The rotated point
    """
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
    """Compute face normal using Newell's method (robust for non-planar faces).

    Args:
        face: List of vertices defining the face

    Returns:
        Normal vector (not normalized)
    """
    normal_x = normal_y = normal_z = 0.0
    face_len = len(face)
    for i in range(face_len):
        v1 = face[i]
        v2 = face[(i + 1) % face_len]
        normal_x += (v1.y - v2.y) * (v1.z + v2.z)
        normal_y += (v1.z - v2.z) * (v1.x + v2.x)
        normal_z += (v1.x - v2.x) * (v1.y + v2.y)
    return Vector3(normal_x, normal_y, normal_z)


def compute_cross_product_normal(face):
    """Compute face normal using cross product method as fallback.

    Args:
        face: List of vertices defining the face

    Returns:
        Normalized normal vector
    """
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
    """Calculate the normal vector for a face.

    Args:
        face: List of vertices defining the face

    Returns:
        Normalized normal vector
    """
    if len(face) < 3:
        return Vector3(0, 0, 1)

    normal = compute_newell_normal(face)

    if normal.magnitude_squared() > 1e-12:
        return normal.normalize()
    else:
        return compute_cross_product_normal(face)


def calculate_face_centroid(face):
    """Calculate the centroid (center point) of a face.

    Args:
        face: List of vertices defining the face

    Returns:
        Centroid as Vector3
    """
    face_len = len(face)
    sum_x = sum_y = sum_z = 0.0
    for vertex in face:
        sum_x += vertex.x
        sum_y += vertex.y
        sum_z += vertex.z
    return Vector3(sum_x / face_len, sum_y / face_len, sum_z / face_len)


# Ray-triangle intersection for shadows
def ray_intersects_triangle(ray_origin, ray_direction, triangle):
    """Test if a ray intersects a triangle using the Möller-Trumbore algorithm.

    Args:
        ray_origin: Origin point of the ray
        ray_direction: Direction vector of the ray
        triangle: List of 3 vertices defining the triangle

    Returns:
        Tuple of (intersects: bool, distance: float)
    """
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
    """Check if a point is occluded from a light source by other objects.

    Args:
        light_pos: Position of the light source
        target_point: Point to check for occlusion
        objects: List of objects that could cast shadows
        current_object: Current object to exclude from shadow testing

    Returns:
        True if the point is occluded, False otherwise
    """
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
    """Clamp a color component to the range [0, 1]."""
    return max(0.0, min(1.0, value))


def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """Convert hex color string to RGB tuple.

    Args:
        hex_color: Hex color string (e.g., "#FFFFFF")

    Returns:
        RGB tuple with values in range [0, 1]
    """
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


# Light sources
class LightSource:
    """Base class for all light sources."""

    def __init__(
        self,
        pos: Vector3,
        color: str,
        brightness: float,
        falloff_type: str = "linear",
        falloff_rate: float = 0.1,
    ):
        """Initialize a light source.

        Args:
            pos: Position of the light
            color: Color as hex string (e.g., "#FFFFFF")
            brightness: Light brightness multiplier
            falloff_type: Type of distance falloff ("none", "linear", "quadratic", "exponential")
            falloff_rate: Rate of falloff with distance
        """
        self.pos = pos
        self.color = hex_to_rgb(color)
        self.color_hex = color
        self.brightness = max(0.0, brightness)
        self.falloff_type = falloff_type
        self.falloff_rate = max(0.001, falloff_rate)

    def calculate_falloff(self, distance: float) -> float:
        """Calculate light falloff based on distance."""
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
        """Calculate lighting contribution at a point. Override in subclasses."""
        return (0.0, 0.0, 0.0)


class AmbientLight(LightSource):
    """Ambient light that illuminates all surfaces equally."""

    def __init__(self, color: str, brightness: float):
        """Initialize ambient light.

        Args:
            color: Color as hex string
            brightness: Light brightness
        """
        super().__init__(Vector3(0, 0, 0), color, brightness, "none", 0.0)

    def calculate_lighting(
        self, point: Vector3, normal: Vector3
    ) -> Tuple[float, float, float]:
        """Calculate ambient lighting (constant for all points)."""
        return (
            self.color[0] * self.brightness,
            self.color[1] * self.brightness,
            self.color[2] * self.brightness,
        )


class PointLight(LightSource):
    """Point light that emanates light in all directions from a single point."""

    def calculate_lighting(
        self, point: Vector3, normal: Vector3
    ) -> Tuple[float, float, float]:
        """Calculate point light contribution using Lambert's cosine law."""
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
    """Directional light that casts parallel rays in a specific direction."""

    def __init__(
        self,
        pos: Vector3,
        direction: Vector3,
        color: str,
        brightness: float,
        falloff_type: str = "linear",
        falloff_rate: float = 0.0,
    ):
        """Initialize directional light.

        Args:
            pos: Position of the light
            direction: Direction the light is pointing
            color: Color as hex string
            brightness: Light brightness
            falloff_type: Type of distance falloff
            falloff_rate: Rate of falloff with distance
        """
        super().__init__(pos, color, brightness, falloff_type, falloff_rate)
        self.direction = direction.normalize()

    def calculate_lighting(
        self, point: Vector3, normal: Vector3
    ) -> Tuple[float, float, float]:
        """Calculate directional light contribution."""
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
    """Spot light that illuminates objects within a cone."""

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
        """Initialize spot light.

        Args:
            pos: Position of the light
            direction: Direction the light is pointing
            color: Color as hex string
            brightness: Light brightness
            cone_angle: Cone angle in radians
            falloff_type: Type of distance falloff
            falloff_rate: Rate of falloff with distance
        """
        super().__init__(pos, color, brightness, falloff_type, falloff_rate)
        self.direction = direction.normalize()
        self.cone_angle = cone_angle

    def calculate_lighting(
        self, point: Vector3, normal: Vector3
    ) -> Tuple[float, float, float]:
        """Calculate spot light contribution within the cone."""
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
    """Light emitted from emissive surfaces."""

    def calculate_lighting(
        self, point: Vector3, normal: Vector3
    ) -> Tuple[float, float, float]:
        """Calculate emissive light contribution."""
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
    """Configuration and management for the lighting system."""

    __slots__ = [
        "light_sources",
        "use_caching",
        "enable_shadows",
        "light_bounces",
        "light_bounce_samples",
        "precompute_bounces",
        "use_advanced_lighting",
        "max_light_distance",
        "max_bounce_distance",
        "shadow_bias",
        "light_contribution_threshold",
        "_lighting_cache",
        "_cache_frame",
        "_temp_light",
        "_bounce_cache",
        "_bounce_cache_valid",
        "_face_lighting_cache",
        "_centroid_normal_cache",
        "_cache_hits",
        "_cache_misses",
        "_precompute_objects",
        "_max_distance_sq",
        "_max_bounce_distance_sq",
        "_precomputed_lighting_cache",
        "_precomputed_lighting_valid",
        "_simple_lighting_result",
    ]

    def __init__(
        self,
        use_advanced_lighting=True,
        max_light_distance=50.0,
        use_caching=True,
        light_bounces=0,
        light_bounce_samples=8,
        precompute_bounces=False,
        max_bounce_distance=8.0,
        light_contribution_threshold=0.001,
        enable_shadows=False,
        shadow_bias=0.001,
    ):
        """Initialize lighting configuration.

        Args:
            use_caching: Enable lighting calculation caching
            enable_shadows: Enable shadow casting
            light_bounces: Number of light bounces for global illumination
            light_bounce_samples: Number of samples per bounce for global illumination
            precompute_bounces: Precompute bounce lighting for performance
            use_advanced_lighting: Enable advanced lighting calculations
            max_light_distance: Maximum distance for light influence
            max_bounce_distance: Maximum distance for bounce lighting
            shadow_bias: Bias to prevent shadow acne
            light_contribution_threshold: Minimum light contribution to consider
        """
        self.light_sources = []
        self.use_caching = use_caching
        self.enable_shadows = enable_shadows
        self.light_bounces = max(0, light_bounces)
        self.light_bounce_samples = max(1, light_bounce_samples)
        self.precompute_bounces = precompute_bounces
        self.use_advanced_lighting = use_advanced_lighting
        self.max_light_distance = max_light_distance
        self.max_bounce_distance = max_bounce_distance
        self.shadow_bias = shadow_bias
        self.light_contribution_threshold = light_contribution_threshold
        self._lighting_cache = {}
        self._cache_frame = 0
        self._temp_light = [0.0, 0.0, 0.0]
        self._bounce_cache = {}
        self._bounce_cache_valid = False
        self._face_lighting_cache = {}
        self._centroid_normal_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._precompute_objects = None
        self._max_distance_sq = max_light_distance * max_light_distance
        self._max_bounce_distance_sq = max_bounce_distance * max_bounce_distance
        self._precomputed_lighting_cache = {}
        self._precomputed_lighting_valid = False
        self._simple_lighting_result = (1.0, 1.0, 1.0)

    def add_light_source(self, light_source):
        """Add a light source to the configuration."""
        self.light_sources.append(light_source)
        if self.use_caching and len(self._lighting_cache) > 1000:
            self._lighting_cache.clear()
            self._face_lighting_cache.clear()
            self._centroid_normal_cache.clear()
        self._bounce_cache_valid = False
        self._precomputed_lighting_valid = False

    def remove_light_source(self, light_source):
        """Remove a light source from the configuration."""
        if light_source in self.light_sources:
            self.light_sources.remove(light_source)
            if self.use_caching:
                self._lighting_cache.clear()
                self._face_lighting_cache.clear()
                self._centroid_normal_cache.clear()
            self._bounce_cache_valid = False
            self._precomputed_lighting_valid = False

    def clear_light_sources(self):
        """Remove all light sources."""
        self.light_sources.clear()
        if self.use_caching:
            self._lighting_cache.clear()
            self._face_lighting_cache.clear()
            self._centroid_normal_cache.clear()
        self._bounce_cache_valid = False
        self._precomputed_lighting_valid = False

    def set_light_bounce_samples(self, samples):
        """Set the number of samples used for bounce lighting calculations.

        Args:
            samples: Number of samples (minimum 1)
        """
        self.light_bounce_samples = max(1, samples)
        if self.use_caching:
            self._lighting_cache.clear()
            self._face_lighting_cache.clear()
        self._bounce_cache_valid = False

    def set_caching(self, enabled):
        """Enable or disable lighting caching."""
        self.use_caching = enabled
        if not enabled:
            self._lighting_cache.clear()
            self._face_lighting_cache.clear()
            self._centroid_normal_cache.clear()

    def set_shadows(self, enabled):
        """Enable or disable shadow casting."""
        self.enable_shadows = enabled
        if self.use_caching:
            self._lighting_cache.clear()
            self._face_lighting_cache.clear()

    def set_advanced_lighting(self, enabled):
        """Enable or disable advanced lighting calculations."""
        self.use_advanced_lighting = enabled
        if self.use_caching:
            self._lighting_cache.clear()
            self._face_lighting_cache.clear()

    def set_light_bounces(self, bounces):
        """Set the number of light bounces for global illumination."""
        self.light_bounces = max(0, bounces)
        if self.use_caching:
            self._lighting_cache.clear()
            self._face_lighting_cache.clear()
        self._bounce_cache_valid = False

    def set_precompute_bounces(self, enabled):
        """Enable or disable bounce lighting precomputation."""
        self.precompute_bounces = enabled
        if enabled:
            self._bounce_cache_valid = False
        else:
            self._bounce_cache.clear()

    def set_precompute_objects(self, objects):
        """Set the objects to use for bounce lighting precomputation."""
        self._precompute_objects = objects
        self._bounce_cache_valid = False

    def _make_cache_key(self, point, normal):
        """Create optimized cache key with fewer float operations."""
        return (
            round(point.x * 200),
            round(point.y * 200),
            round(point.z * 200),
            round(normal.x * 200),
            round(normal.y * 200),
            round(normal.z * 200),
        )

    def _get_cached_centroid_normal(self, obj, face_idx):
        """Get cached face centroid and normal data."""
        if not self.use_caching:
            return obj._compute_face_data(face_idx)

        cache_key = (id(obj), face_idx, obj._object_version)
        cached = self._centroid_normal_cache.get(cache_key)
        if cached is not None:
            return cached

        centroid, normal = obj._compute_face_data(face_idx)
        if len(self._centroid_normal_cache) < 2000:
            self._centroid_normal_cache[cache_key] = (centroid, normal)
        return centroid, normal

    def _get_cached_face_lighting(self, obj, face_idx, centroid, normal):
        """Get cached face lighting data."""
        if not self.use_caching:
            return self._compute_face_lighting(centroid, normal)

        cache_key = (
            id(obj),
            face_idx,
            obj._object_version,
            *self._make_cache_key(centroid, Vector3(0, 0, 0))[:3],
        )

        cached = self._face_lighting_cache.get(cache_key)
        if cached is not None:
            self._cache_hits += 1
            return cached

        self._cache_misses += 1
        face_lighting = self._compute_face_lighting(centroid, normal)
        if len(self._face_lighting_cache) < 2000:
            self._face_lighting_cache[cache_key] = face_lighting
        return face_lighting

    def _compute_face_lighting(self, centroid, normal):
        """Compute direct lighting for a face."""
        face_lighting = [0.0, 0.0, 0.0]
        for light_source in self.light_sources:
            if isinstance(light_source, AmbientLight):
                ambient_contrib = light_source.calculate_lighting(centroid, normal)
                face_lighting[0] += ambient_contrib[0]
                face_lighting[1] += ambient_contrib[1]
                face_lighting[2] += ambient_contrib[2]
            else:
                dx = light_source.pos.x - centroid.x
                dy = light_source.pos.y - centroid.y
                dz = light_source.pos.z - centroid.z
                light_distance_sq = dx * dx + dy * dy + dz * dz

                if light_distance_sq > self._max_distance_sq:
                    continue

                light_contribution = light_source.calculate_lighting(centroid, normal)

                if (
                    light_contribution[0]
                    + light_contribution[1]
                    + light_contribution[2]
                ) < self.light_contribution_threshold:
                    continue

                face_lighting[0] += light_contribution[0]
                face_lighting[1] += light_contribution[1]
                face_lighting[2] += light_contribution[2]
        return face_lighting

    def precompute_bounce_lighting(self, objects=None):
        """Precompute bounce lighting for performance optimization."""
        if not self.precompute_bounces or self.light_bounces == 0:
            return

        if objects is None:
            objects = self._precompute_objects

        if objects is None:
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

        processed = 0
        max_processed = 500

        for obj in objects:
            for face_idx in range(len(obj.faces)):
                if processed >= max_processed:
                    break

                centroid, normal = self._get_cached_centroid_normal(obj, face_idx)

                bounce_strength = 0.25
                total_bounce_lighting = [0.0, 0.0, 0.0]

                for bounce in range(min(self.light_bounces, 2)):
                    bounce_strength *= 0.8

                    if bounce_strength < 0.02:
                        break

                    bounce_lighting = self._calculate_bounce_lighting_optimized(
                        centroid,
                        normal,
                        objects,
                        obj,
                        face_idx,
                        max_distance=self.max_bounce_distance,
                        bounce_strength=bounce_strength,
                    )

                    bounce_total = (
                        bounce_lighting[0] + bounce_lighting[1] + bounce_lighting[2]
                    )
                    if bounce_total < self.light_contribution_threshold:
                        break

                    total_bounce_lighting[0] += bounce_lighting[0]
                    total_bounce_lighting[1] += bounce_lighting[1]
                    total_bounce_lighting[2] += bounce_lighting[2]

                cache_key = (
                    round(centroid.x * 100),
                    round(centroid.y * 100),
                    round(centroid.z * 100),
                    round(normal.x * 100),
                    round(normal.y * 100),
                    round(normal.z * 100),
                )

                self._bounce_cache[cache_key] = total_bounce_lighting
                processed += 1
            if processed >= max_processed:
                break

        self._bounce_cache_valid = True

    def precompute_direct_lighting(self, objects=None):
        """Precompute all lighting calculations for better first-frame performance."""
        if objects is None:
            objects = self._precompute_objects

        if objects is None:
            return

        self._precomputed_lighting_cache.clear()

        if isinstance(objects, Object):
            objects = [objects]

        flat_objects = []
        for obj_list in objects:
            if isinstance(obj_list, list):
                flat_objects.extend(obj_list)
            else:
                flat_objects.append(obj_list)
        objects = flat_objects

        processed = 0
        max_processed = 1000

        for obj in objects:
            for face_idx in range(len(obj.faces)):
                if processed >= max_processed:
                    break

                centroid, normal = self._get_cached_centroid_normal(obj, face_idx)

                lighting = self._compute_lighting(
                    centroid, normal, objects, obj, face_idx
                )

                cache_key = self._make_cache_key(centroid, normal)
                self._precomputed_lighting_cache[cache_key] = lighting
                processed += 1

            if processed >= max_processed:
                break

        self._precomputed_lighting_valid = True

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
        """Calculate optimized bounce lighting using configurable sample count."""
        bounce_lighting = [0.0, 0.0, 0.0]

        if bounce_strength < 0.02:
            return bounce_lighting

        num_samples = self.light_bounce_samples
        max_faces_per_object = 8
        sample_weight = 1.0 / num_samples

        up = (
            Vector3(0, 1, 0)
            if abs(normal.dot(Vector3(0, 1, 0))) <= 0.9
            else Vector3(1, 0, 0)
        )
        tangent = normal.cross(up).normalize()
        bitangent = normal.cross(tangent).normalize()

        two_pi = 2 * pi
        offset_amount = 0.4

        for i in range(num_samples):
            angle = (i / num_samples) * two_pi
            cos_angle = cos(angle)
            sin_angle = sin(angle)

            offset_dir_x = (
                tangent.x * cos_angle * offset_amount
                + bitangent.x * sin_angle * offset_amount
            )
            offset_dir_y = (
                tangent.y * cos_angle * offset_amount
                + bitangent.y * sin_angle * offset_amount
            )
            offset_dir_z = (
                tangent.z * cos_angle * offset_amount
                + bitangent.z * sin_angle * offset_amount
            )

            sample_dir = Vector3(
                normal.x + offset_dir_x,
                normal.y + offset_dir_y,
                normal.z + offset_dir_z,
            ).normalize()

            for obj in objects:
                if obj is current_object:
                    continue

                faces_checked = 0
                face_count = len(obj.faces)
                step = max(1, face_count // max_faces_per_object)

                for face_idx in range(0, face_count, step):
                    if faces_checked >= max_faces_per_object:
                        break
                    faces_checked += 1

                    face_center, face_normal = self._get_cached_centroid_normal(
                        obj, face_idx
                    )

                    dx = face_center.x - point.x
                    dy = face_center.y - point.y
                    dz = face_center.z - point.z
                    distance_sq = dx * dx + dy * dy + dz * dz

                    if distance_sq > self._max_bounce_distance_sq or distance_sq < 0.01:
                        continue

                    distance = sqrt(distance_sq)
                    inv_distance = 1.0 / distance
                    direction_to_face = Vector3(
                        dx * inv_distance, dy * inv_distance, dz * inv_distance
                    )

                    if normal.dot(direction_to_face) < 0.1:
                        continue
                    if (
                        face_normal.dot(
                            Vector3(
                                -direction_to_face.x,
                                -direction_to_face.y,
                                -direction_to_face.z,
                            )
                        )
                        < 0.1
                    ):
                        continue

                    sample_dot = sample_dir.dot(direction_to_face)
                    if sample_dot < 0.6:
                        continue

                    face_lighting = self._get_cached_face_lighting(
                        obj, face_idx, face_center, face_normal
                    )

                    face_lighting_total = (
                        face_lighting[0] + face_lighting[1] + face_lighting[2]
                    )
                    if face_lighting_total < self.light_contribution_threshold:
                        continue

                    falloff = max(0.0, 1.0 - (distance / max_distance))
                    angle_factor = max(0.0, normal.dot(direction_to_face))
                    surface_angle_factor = max(
                        0.0,
                        face_normal.dot(
                            Vector3(
                                -direction_to_face.x,
                                -direction_to_face.y,
                                -direction_to_face.z,
                            )
                        ),
                    )

                    bounce_factor = (
                        bounce_strength
                        * falloff
                        * angle_factor
                        * surface_angle_factor
                        * 0.25
                    )

                    if bounce_factor < 0.02:
                        continue

                    bounce_lighting[0] += face_lighting[0] * bounce_factor
                    bounce_lighting[1] += face_lighting[1] * bounce_factor
                    bounce_lighting[2] += face_lighting[2] * bounce_factor

        bounce_lighting[0] *= sample_weight
        bounce_lighting[1] *= sample_weight
        bounce_lighting[2] *= sample_weight

        return (
            max(0.0, min(1.0, bounce_lighting[0])),
            max(0.0, min(1.0, bounce_lighting[1])),
            max(0.0, min(1.0, bounce_lighting[2])),
        )

    def _calculate_realtime_bounce_lighting(
        self,
        point,
        normal,
        objects,
        current_object,
        current_face_idx,
    ):
        bounce_lighting = [0.0, 0.0, 0.0]

        if self.light_bounces == 0:
            return bounce_lighting

        bounce_strength = 0.25
        max_distance = self.max_bounce_distance
        num_samples = self.light_bounce_samples
        max_faces_per_object = 8

        for bounce in range(self.light_bounces):
            bounce_strength *= 0.8
            if bounce_strength < 0.02:
                break

            sample_weight = 1.0 / num_samples
            up = (
                Vector3(0, 1, 0)
                if abs(normal.dot(Vector3(0, 1, 0))) <= 0.9
                else Vector3(1, 0, 0)
            )
            tangent = normal.cross(up).normalize()
            bitangent = normal.cross(tangent).normalize()

            two_pi = 2 * pi
            offset_amount = 0.4

            for i in range(num_samples):
                angle = (i / num_samples) * two_pi
                cos_angle = cos(angle)
                sin_angle = sin(angle)

                offset_dir_x = (
                    tangent.x * cos_angle * offset_amount
                    + bitangent.x * sin_angle * offset_amount
                )
                offset_dir_y = (
                    tangent.y * cos_angle * offset_amount
                    + bitangent.y * sin_angle * offset_amount
                )
                offset_dir_z = (
                    tangent.z * cos_angle * offset_amount
                    + bitangent.z * sin_angle * offset_amount
                )

                sample_dir = Vector3(
                    normal.x + offset_dir_x,
                    normal.y + offset_dir_y,
                    normal.z + offset_dir_z,
                ).normalize()

                for obj in objects:
                    if obj is current_object:
                        continue

                    faces_checked = 0
                    face_count = len(obj.faces)
                    step = max(1, face_count // max_faces_per_object)

                    for face_idx in range(0, face_count, step):
                        if faces_checked >= max_faces_per_object:
                            break
                        faces_checked += 1

                        face_center, face_normal = self._get_cached_centroid_normal(
                            obj, face_idx
                        )

                        dx = face_center.x - point.x
                        dy = face_center.y - point.y
                        dz = face_center.z - point.z
                        distance_sq = dx * dx + dy * dy + dz * dz

                        if (
                            distance_sq > self._max_bounce_distance_sq
                            or distance_sq < 0.01
                        ):
                            continue

                        distance = sqrt(distance_sq)
                        inv_distance = 1.0 / distance
                        direction_to_face = Vector3(
                            dx * inv_distance, dy * inv_distance, dz * inv_distance
                        )

                        if normal.dot(direction_to_face) < 0.1:
                            continue
                        if (
                            face_normal.dot(
                                Vector3(
                                    -direction_to_face.x,
                                    -direction_to_face.y,
                                    -direction_to_face.z,
                                )
                            )
                            < 0.1
                        ):
                            continue

                        sample_dot = sample_dir.dot(direction_to_face)
                        if sample_dot < 0.6:
                            continue

                        face_lighting = self._get_cached_face_lighting(
                            obj, face_idx, face_center, face_normal
                        )

                        face_lighting_total = (
                            face_lighting[0] + face_lighting[1] + face_lighting[2]
                        )
                        if face_lighting_total < self.light_contribution_threshold:
                            continue

                        falloff = max(0.0, 1.0 - (distance / max_distance))
                        angle_factor = max(0.0, normal.dot(direction_to_face))
                        surface_angle_factor = max(
                            0.0,
                            face_normal.dot(
                                Vector3(
                                    -direction_to_face.x,
                                    -direction_to_face.y,
                                    -direction_to_face.z,
                                )
                            ),
                        )

                        bounce_factor = (
                            bounce_strength
                            * falloff
                            * angle_factor
                            * surface_angle_factor
                            * 0.25
                        )

                        if bounce_factor < 0.02:
                            continue

                        bounce_lighting[0] += face_lighting[0] * bounce_factor
                        bounce_lighting[1] += face_lighting[1] * bounce_factor
                        bounce_lighting[2] += face_lighting[2] * bounce_factor

            bounce_lighting[0] *= sample_weight
            bounce_lighting[1] *= sample_weight
            bounce_lighting[2] *= sample_weight

        return (
            max(0.0, min(1.0, bounce_lighting[0])),
            max(0.0, min(1.0, bounce_lighting[1])),
            max(0.0, min(1.0, bounce_lighting[2])),
        )

    def get_precomputed_bounce_lighting(self, point, normal):
        """Get precomputed bounce lighting for a point."""
        if not self.precompute_bounces or not self._bounce_cache_valid:
            return [0.0, 0.0, 0.0]

        cache_key = (
            round(point.x * 100),
            round(point.y * 100),
            round(point.z * 100),
            round(normal.x * 100),
            round(normal.y * 100),
            round(normal.z * 100),
        )

        return self._bounce_cache.get(cache_key, [0.0, 0.0, 0.0])

    def get_precomputed_lighting(self, point, normal):
        """Get precomputed lighting for a point if available."""
        if not self._precomputed_lighting_valid:
            return None

        cache_key = self._make_cache_key(point, normal)
        return self._precomputed_lighting_cache.get(cache_key)

    def calculate_lighting_at_point(
        self, point, normal, objects=None, current_object=None, current_face_idx=None
    ):
        """Calculate total lighting (direct + indirect) at a point."""
        if not self.use_advanced_lighting:
            return self._simple_lighting_result

        precomputed = self.get_precomputed_lighting(point, normal)
        if precomputed is not None:
            return precomputed

        if not self.use_caching:
            return self._compute_lighting(
                point, normal, objects, current_object, current_face_idx
            )

        cache_key = self._make_cache_key(point, normal)
        cached_result = self._lighting_cache.get(cache_key)
        if cached_result is not None:
            self._cache_hits += 1
            return cached_result

        self._cache_misses += 1
        result = self._compute_lighting(
            point, normal, objects, current_object, current_face_idx
        )
        if len(self._lighting_cache) < 2000:
            self._lighting_cache[cache_key] = result
        return result

    def _compute_lighting(
        self, point, normal, objects=None, current_object=None, current_face_idx=None
    ):
        """Compute direct and indirect lighting at a point."""
        self._temp_light[0] = 0.0
        self._temp_light[1] = 0.0
        self._temp_light[2] = 0.0

        for light_source in self.light_sources:
            if isinstance(light_source, AmbientLight):
                light_contribution = light_source.calculate_lighting(point, normal)
                self._temp_light[0] += light_contribution[0]
                self._temp_light[1] += light_contribution[1]
                self._temp_light[2] += light_contribution[2]
            else:
                dx = light_source.pos.x - point.x
                dy = light_source.pos.y - point.y
                dz = light_source.pos.z - point.z
                distance_sq = dx * dx + dy * dy + dz * dz

                if distance_sq > self._max_distance_sq:
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
                contribution_total = (
                    light_contribution[0]
                    + light_contribution[1]
                    + light_contribution[2]
                )

                if contribution_total < self.light_contribution_threshold:
                    continue

                self._temp_light[0] += light_contribution[0]
                self._temp_light[1] += light_contribution[1]
                self._temp_light[2] += light_contribution[2]

        if self.light_bounces > 0 and objects:
            if self.precompute_bounces and self._bounce_cache_valid:
                bounce_lighting = self.get_precomputed_bounce_lighting(point, normal)
                self._temp_light[0] += bounce_lighting[0]
                self._temp_light[1] += bounce_lighting[1]
                self._temp_light[2] += bounce_lighting[2]
            else:
                bounce_lighting = self._calculate_realtime_bounce_lighting(
                    point, normal, objects, current_object, current_face_idx
                )
                self._temp_light[0] += bounce_lighting[0]
                self._temp_light[1] += bounce_lighting[1]
                self._temp_light[2] += bounce_lighting[2]

        return (
            max(0.0, min(1.0, self._temp_light[0])),
            max(0.0, min(1.0, self._temp_light[1])),
            max(0.0, min(1.0, self._temp_light[2])),
        )

    def get_cache_stats(self):
        """Get cache performance statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "lighting_cache_size": len(self._lighting_cache),
            "face_lighting_cache_size": len(self._face_lighting_cache),
            "centroid_normal_cache_size": len(self._centroid_normal_cache),
            "bounce_cache_size": len(self._bounce_cache),
            "bounce_cache_valid": self._bounce_cache_valid,
            "precomputed_cache_size": len(self._precomputed_lighting_cache),
            "precomputed_valid": self._precomputed_lighting_valid,
        }


def set_as_light_source(obj: Dict, brightness: float, color: str = "#ffffff"):
    """Set an object as a light source (deprecated - use Object.set_as_light_source)."""
    obj["is_light_source"] = True
    obj["emission_brightness"] = max(0.0, brightness)
    obj["emission_color_hex"] = color

    r, g, b = hex_to_rgb(color)
    obj["emission_color"] = (r, g, b)


def update_emissive_lights(objects, lighting_config):
    """Update emissive lights from objects marked as light sources."""
    if isinstance(objects, Object):
        objects = [objects]

    emissive_lights = [
        light
        for light in lighting_config.light_sources
        if isinstance(light, EmissiveLight)
    ]
    for light in emissive_lights:
        lighting_config.remove_light_source(light)

    light_count = 0
    max_emissive_lights = 20

    for obj in objects:
        if obj.is_light_source and light_count < max_emissive_lights:
            emission_brightness = obj.emission_brightness
            emission_color_hex = obj.emission_color_hex

            face_step = max(1, len(obj.faces) // 5)

            for face_idx in range(0, len(obj.faces), face_step):
                if light_count >= max_emissive_lights:
                    break

                face, material = obj.faces[face_idx]
                face_center = sum(face, Vector3(0, 0, 0)) / len(face)
                emissive_light = EmissiveLight(
                    pos=face_center,
                    color=emission_color_hex,
                    brightness=emission_brightness,
                    falloff_type="linear",
                    falloff_rate=0.08,
                )

                lighting_config.add_light_source(emissive_light)
                light_count += 1


# Camera and rendering
class Camera:
    """3D camera for rendering scenes with lighting and projection."""

    __slots__ = [
        "pos",
        "direction",
        "lighting_config",
        "use_caching",
        "pen",
        "_render_items",
        "_temp_projected",
        "_screen_half_w",
        "_screen_half_h",
        "_fov_factor",
        "_aspect_ratio",
        "_projection_cache",
        "_visibility_cache",
        "_cache_frame",
        "_max_render_items",
    ]

    def __init__(self, pos, direction=None, lighting_config=None, use_caching=True):
        """Initialize the camera with performance optimizations.

        Args:
            pos: Camera position as Vector3
            direction: Camera direction as Vector3 (defaults to forward)
            lighting_config: LightingConfig instance
            use_caching: Enable caching for performance
        """
        self.pos = pos
        if direction is None:
            self.direction = Vector3(0, 0, 1)
        else:
            self.direction = direction.normalize()
        self.lighting_config = lighting_config or LightingConfig()
        self.use_caching = use_caching

        try:
            turtle.tracer(0)
            turtle.setup(800, 600)
            pen = turtle.Turtle()
            pen.hideturtle()
            pen.penup()
            pen.color("black")
            pen.speed(0)
            self.pen = pen
        except Exception as e:
            print(f"Warning: Error initializing turtle graphics: {str(e)}")
            self.pen = None

        self._render_items = []
        self._temp_projected = []
        self._screen_half_w = 400.0
        self._screen_half_h = 300.0
        self._fov_factor = 1.0
        self._aspect_ratio = 1.0
        self._projection_cache = {}
        self._visibility_cache = {}
        self._cache_frame = 0
        self._max_render_items = 2000

    def set_lighting_config(self, lighting_config):
        """Set the lighting configuration."""
        self.lighting_config = lighting_config

    def set_caching(self, enabled):
        """Enable or disable caching."""
        self.use_caching = enabled
        if not enabled:
            self._projection_cache.clear()
            self._visibility_cache.clear()

    def move_axis(self, pos):
        """Move the camera by a vector amount."""
        self.pos += pos
        if self.use_caching:
            self._projection_cache.clear()
            self._visibility_cache.clear()
        return self.pos

    def move(self, steps, horizontal_only=False):
        """Move the camera forward/backward.

        Args:
            steps: Distance to move (positive = forward)
            horizontal_only: Only move in XZ plane
        """
        if horizontal_only:
            forward = Vector3(self.direction.x, 0, self.direction.z).normalize()
        else:
            forward = self.direction

        move = forward * steps
        self.pos += move
        if self.use_caching:
            self._projection_cache.clear()
            self._visibility_cache.clear()
        return self.pos

    def strafe(self, steps):
        """Move the camera left/right relative to view direction."""
        up = Vector3(0, 1, 0)
        right = self.direction.cross(up).normalize()
        move = right * steps
        self.pos += move
        if self.use_caching:
            self._projection_cache.clear()
            self._visibility_cache.clear()
        return self.pos

    def move_relative(self, pos, horizontal_only=False):
        """Move camera relative to its current orientation."""
        self.move(pos.z, horizontal_only)
        self.strafe(pos.x)
        self.pos.y -= pos.y

    def rotate(self, pitch_delta, yaw_delta):
        """Rotate the camera by pitch and yaw angles."""
        up = Vector3(0, 1, 0)

        if yaw_delta != 0:
            cos_yaw = cos(yaw_delta)
            sin_yaw = sin(yaw_delta)
            new_x = self.direction.x * cos_yaw - self.direction.z * sin_yaw
            new_z = self.direction.x * sin_yaw + self.direction.z * cos_yaw
            self.direction = Vector3(new_x, self.direction.y, new_z).normalize()

        if pitch_delta != 0:
            cos_pitch = cos(pitch_delta)
            sin_pitch = sin(pitch_delta)
            new_direction = self.direction * cos_pitch + up * sin_pitch
            self.direction = new_direction.normalize()

        if self.use_caching and (pitch_delta != 0 or yaw_delta != 0):
            self._projection_cache.clear()
            self._visibility_cache.clear()

    def get_view_direction(self):
        """Get the current view direction."""
        return self.direction

    def _get_cached_projection(self, point, screen_width, screen_height, fov):
        if not self.use_caching:
            return self.project_point(point, screen_width, screen_height, fov)

        cache_key = (
            int(point.x * 1000),
            int(point.y * 1000),
            int(point.z * 1000),
            int(self.pos.x * 1000),
            int(self.pos.y * 1000),
            int(self.pos.z * 1000),
            int(self.direction.x * 1000),
            int(self.direction.y * 1000),
            int(self.direction.z * 1000),
            screen_width,
            screen_height,
            fov,
        )

        cached = self._projection_cache.get(cache_key)
        if cached is not None:
            return cached

        result = self.project_point(point, screen_width, screen_height, fov)
        if len(self._projection_cache) < 5000:
            self._projection_cache[cache_key] = result
        return result

    def _get_cached_visibility(self, face_center, normal):
        if not self.use_caching:
            return self.is_face_visible(face_center, normal)

        cache_key = (
            int(face_center.x * 1000),
            int(face_center.y * 1000),
            int(face_center.z * 1000),
            int(normal.x * 1000),
            int(normal.y * 1000),
            int(normal.z * 1000),
            int(self.pos.x * 1000),
            int(self.pos.y * 1000),
            int(self.pos.z * 1000),
        )

        cached = self._visibility_cache.get(cache_key)
        if cached is not None:
            return cached

        result = self.is_face_visible(face_center, normal)
        if len(self._visibility_cache) < 3000:
            self._visibility_cache[cache_key] = result
        return result

    def project_point(self, point, screen_width=800, screen_height=600, fov=90):
        """Project a 3D point to 2D screen coordinates.

        Args:
            point: 3D point to project
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            fov: Field of view in degrees

        Returns:
            Tuple of (x, y) screen coordinates or None if behind camera
        """
        relative_x = point.x - self.pos.x
        relative_y = point.y - self.pos.y
        relative_z = point.z - self.pos.z

        forward = self.direction
        up = Vector3(0, 1, 0)

        right_x = forward.y * up.z - forward.z * up.y
        right_y = forward.z * up.x - forward.x * up.z
        right_z = forward.x * up.y - forward.y * up.x

        right_mag = sqrt(right_x * right_x + right_y * right_y + right_z * right_z)
        if right_mag == 0:
            return None
        right_x /= right_mag
        right_y /= right_mag
        right_z /= right_mag

        actual_up_x = right_y * forward.z - right_z * forward.y
        actual_up_y = right_z * forward.x - right_x * forward.z
        actual_up_z = right_x * forward.y - right_y * forward.x

        x = relative_x * right_x + relative_y * right_y + relative_z * right_z
        y = (
            relative_x * actual_up_x
            + relative_y * actual_up_y
            + relative_z * actual_up_z
        )
        z = relative_x * forward.x + relative_y * forward.y + relative_z * forward.z

        if z <= 0:
            return None

        fov_rad = radians(fov)
        f = 1.0 / tan(fov_rad * 0.5)

        screen_x = (x * f / z) * (screen_height / screen_width)
        screen_y = y * f / z

        screen_x = screen_x * (screen_width * 0.5) + screen_width * 0.5
        screen_y = screen_y * (screen_height * 0.5) + screen_height * 0.5

        turtle_x = screen_x - screen_width * 0.5
        turtle_y = screen_y - screen_height * 0.5

        return (turtle_x, turtle_y)

    def is_face_visible(self, face_center, normal):
        """Check if a face is visible from the camera (backface culling)."""
        view_x = face_center.x - self.pos.x
        view_y = face_center.y - self.pos.y
        view_z = face_center.z - self.pos.z
        view_mag = sqrt(view_x * view_x + view_y * view_y + view_z * view_z)
        if view_mag == 0:
            return False
        view_x /= view_mag
        view_y /= view_mag
        view_z /= view_mag
        dot_product = normal.x * view_x + normal.y * view_y + normal.z * view_z
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
        draw_light_sources=False,
    ):
        """Render the scene with improved performance and error handling.

        Args:
            objects: List of objects to render
            materials: Dictionary of materials or list of Material objects
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            fov: Field of view in degrees
            show_normals: Draw face normals for debugging
            cull_back_faces: Enable backface culling
            draw_light_sources: Draw light source positions
        """
        if self.pen is None:
            print("Warning: Turtle graphics not available, skipping render")
            return

        try:
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

            if not self.lighting_config._precomputed_lighting_valid:
                if self.lighting_config._precompute_objects is None:
                    self.lighting_config.set_precompute_objects(objects)
                self.lighting_config.precompute_direct_lighting(objects)

            if (
                self.lighting_config.precompute_bounces
                and not self.lighting_config._bounce_cache_valid
                and self.lighting_config.light_bounces > 0
            ):
                if self.lighting_config._precompute_objects is None:
                    self.lighting_config.set_precompute_objects(objects)
                self.lighting_config.precompute_bounce_lighting(objects)

            self._render_items.clear()
            self._cache_frame += 1

            if self.use_caching and self._cache_frame % 100 == 0:
                if len(self._projection_cache) > 7000:
                    self._projection_cache.clear()
                if len(self._visibility_cache) > 4000:
                    self._visibility_cache.clear()

            visible_faces = 0
            max_light_distance_sq = (
                self.lighting_config.max_light_distance
                * self.lighting_config.max_light_distance
            )

            for object in objects:
                if (
                    not object.faces
                    or len(self._render_items) >= self._max_render_items
                ):
                    continue

                if object.faces:
                    object_distance_sq = (
                        (object.faces[0][0][0].x - self.pos.x) ** 2
                        + (object.faces[0][0][0].y - self.pos.y) ** 2
                        + (object.faces[0][0][0].z - self.pos.z) ** 2
                    )
                    if object_distance_sq > max_light_distance_sq * 4:
                        continue

                is_light_source = object.is_light_source
                emission_brightness = object.emission_brightness
                emission_color = object.emission_color

                for face_idx, (face, material) in enumerate(object.faces):
                    if len(self._render_items) >= self._max_render_items:
                        break

                    if material:
                        material = f"{object.name}.{material}"

                    centroid, normal = object.get_face_data(face_idx)

                    distance_sq = (
                        (centroid.x - self.pos.x) ** 2
                        + (centroid.y - self.pos.y) ** 2
                        + (centroid.z - self.pos.z) ** 2
                    )
                    if distance_sq > max_light_distance_sq:
                        continue

                    distance = sqrt(distance_sq)

                    if cull_back_faces and not self._get_cached_visibility(
                        centroid, normal
                    ):
                        continue

                    visible_faces += 1

                    self._temp_projected.clear()
                    all_visible = True
                    for point in face:
                        proj = self._get_cached_projection(
                            point, screen_width, screen_height, fov
                        )
                        if proj is None:
                            all_visible = False
                            break
                        self._temp_projected.append(proj)

                    if not all_visible:
                        continue

                    lighting = self.lighting_config.calculate_lighting_at_point(
                        centroid, normal, objects, object, face_idx
                    )

                    if material in materials:
                        base_color = materials[material]
                        color = [
                            max(0.0, min(1.0, base_color[0] * lighting[0])),
                            max(0.0, min(1.0, base_color[1] * lighting[1])),
                            max(0.0, min(1.0, base_color[2] * lighting[2])),
                        ]
                    else:
                        avg_light = (lighting[0] + lighting[1] + lighting[2]) / 3.0
                        color = [avg_light, avg_light, avg_light]

                    if is_light_source:
                        color[0] = min(
                            1.0, color[0] + emission_color[0] * emission_brightness
                        )
                        color[1] = min(
                            1.0, color[1] + emission_color[1] * emission_brightness
                        )
                        color[2] = min(
                            1.0, color[2] + emission_color[2] * emission_brightness
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
                        normal_start_proj = self._get_cached_projection(
                            centroid, screen_width, screen_height, fov
                        )
                        normal_end_proj = self._get_cached_projection(
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

                    light_distance_sq = (
                        (light_source.pos.x - self.pos.x) ** 2
                        + (light_source.pos.y - self.pos.y) ** 2
                        + (light_source.pos.z - self.pos.z) ** 2
                    )
                    if light_distance_sq > max_light_distance_sq:
                        continue

                    light_proj = self._get_cached_projection(
                        light_source.pos, screen_width, screen_height, fov
                    )
                    if light_proj:
                        light_distance = sqrt(light_distance_sq)
                        self._render_items.append(
                            {
                                "type": "light_source",
                                "position": light_proj,
                                "color": [
                                    max(0.0, min(1.0, color * light_source.brightness))
                                    for color in light_source.color
                                ],
                                "distance": light_distance,
                            }
                        )

            self._render_items.sort(key=lambda x: x["distance"], reverse=True)

            for item in self._render_items:
                try:
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

                except Exception as e:
                    print(f"Warning: Error rendering item: {str(e)}")
                    continue

        except Exception as e:
            print(f"Error during rendering: {str(e)}")


def accurate_sleep(seconds: int | float):
    """Function to sleep accurately.

    Args:
        seconds (int | float): Time to sleep
    """
    if seconds == 0:
        return
    elif seconds < 0.05:
        target = time.perf_counter() + seconds
        while time.perf_counter() < target:
            pass
    else:
        time.sleep(seconds)


total = []
frame = 0
sample_freq = 10
fps_data = []
frames = []


def normalize_framerate(target):
    """Decorator function to normalize a function's runtime.

    Args:
        target (int): Framerate to normalize to
    """

    def decorator(func):
        def wrapped(*args, **kwargs):
            global total, frame, fps_data, frames
            frame += 1
            start_time = time.time()
            result = func(*args, **kwargs)
            total.append(time.time() - start_time)

            if frame % sample_freq == 0:
                avg_fps = 1 / (sum(total) / len(total))
                fps_data.append(avg_fps)
                frames.append(frame // sample_freq)

                print(f"FPS: {avg_fps:.2f}")
                total = []

            time_to_sleep = max(0, (1 / target) - (time.time() - start_time))
            accurate_sleep(time_to_sleep)
            return result

        return wrapped

    return decorator


def mouse_init():
    """Initiates the mouse"""
    global mouse_delta_x, mouse_delta_y, last_mouse_x, last_mouse_y, mouse_initialized, mouse_listener
    mouse_delta_x = 0
    mouse_delta_y = 0
    last_mouse_x = 0
    last_mouse_y = 0
    mouse_initialized = False
    if not using_mouse:
        mouse_listener = MouseListener(on_move=on_mouse_move)


def on_mouse_move(x, y):
    """Mouse listener. Use `handle_movement` instead for movement handling.

    Args:
        x (int): Last mouse x
        y (int): Last mouse y
    """
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
    """Handles camera movement with improved error handling

    Args:
        speed (float, optional): Speed of the camera. Defaults to 0.2.
        sensitivity (float, optional): Sensitivity of the camera. Defaults to 0.05.

    Returns:
        Vector3: Camera move axis (in relation to the camera)
        Vector3: Camera rotate angle
    """
    global mouse_delta_x, mouse_delta_y, using_mouse

    camera_movement = zero3()
    camera_angle = zero2()

    # Add try-catch for keyboard input to prevent crashes
    try:
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
    except Exception:
        # Graceful fallback if keyboard module fails
        pass

    if mouse_locked:
        try:
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
        except Exception:
            # Handle mouse errors gracefully
            pass

    return camera_movement, camera_angle


# Load files n stuff
class Material:
    """Material class"""

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
    """3D object class that holds faces and materials with caching support.

    This class represents a 3D object composed of faces (polygons) and their associated
    materials. It supports lighting calculations, transformations, and performance
    optimizations through caching.

    Attributes:
        name (str): Name identifier for the object
        faces (List): List of tuples containing (face_vertices, material_name)
        is_light_source (bool): Whether this object emits light
        emission_brightness (float): Brightness of emitted light if light source
        emission_color_hex (str): Hex color of emitted light
        emission_color (tuple): RGB color tuple of emitted light
        use_caching (bool): Enable caching for performance optimization
    """

    __slots__ = [
        "name",
        "faces",
        "is_light_source",
        "emission_brightness",
        "emission_color_hex",
        "emission_color",
        "use_caching",
        "_face_cache",
        "_object_version",
    ]

    def __init__(self, name: str, faces: List = None, use_caching=True):
        """Initialize a 3D object.

        Args:
            name (str): Name identifier for the object
            faces (List, optional): List of face data. Defaults to None (empty list)
            use_caching (bool, optional): Enable face data caching. Defaults to True
        """
        self.name = name
        self.faces = faces or []
        self.is_light_source = False
        self.emission_brightness = 0.0
        self.emission_color_hex = "#ffffff"
        self.emission_color = (1.0, 1.0, 1.0)
        self.use_caching = use_caching
        self._face_cache = {}
        self._object_version = 0

    def set_caching(self, enabled: bool):
        """Enable or disable face data caching.

        Args:
            enabled (bool): Whether to enable caching for face calculations
        """
        self.use_caching = enabled
        if not enabled:
            self._face_cache.clear()

    def _invalidate_cache(self):
        """Invalidate all cached face data by incrementing version and clearing cache.

        Called automatically when object geometry changes to ensure cache consistency.
        """
        if self.use_caching:
            self._object_version += 1
            self._face_cache.clear()

    def set_as_light_source(self, brightness: float, color: str = "#ffffff"):
        """Configure this object as a light-emitting source.

        Args:
            brightness (float): Light emission brightness (minimum 0.0)
            color (str, optional): Hex color of emitted light. Defaults to "#ffffff"
        """
        self.is_light_source = True
        self.emission_brightness = max(0.0, brightness)
        self.emission_color_hex = color
        self.emission_color = hex_to_rgb(color)
        self._invalidate_cache()

    def precompute_face_cache(self):
        """Precomputes the face cache"""
        if not self.use_caching:
            return

        self._face_cache.clear()
        for face_idx in range(len(self.faces)):
            cache_key = (face_idx, self._object_version)
            if cache_key not in self._face_cache:
                result = self._compute_face_data(face_idx)
                self._face_cache[cache_key] = result

    def rotate(self, anchor: Vector3, rotation_vector: Vector3):
        """Rotate the entire object around an arbitrary axis.

        Args:
            anchor (Vector3): Point around which to rotate
            rotation_vector (Vector3): Rotation axis and angle (magnitude = angle in radians)
        """
        angle = sqrt(rotation_vector.x**2 + rotation_vector.y**2 + rotation_vector.z**2)

        if angle == 0:
            return

        inv_angle = 1.0 / angle
        axis = Vector3(
            rotation_vector.x * inv_angle,
            rotation_vector.y * inv_angle,
            rotation_vector.z * inv_angle,
        )

        for face_idx, (face, material) in enumerate(self.faces):
            rotated_face = [
                vertex.rotate_point_around_axis(anchor, axis, angle) for vertex in face
            ]
            self.faces[face_idx] = (rotated_face, material)

        self._invalidate_cache()

    def move(self, axis: Vector3):
        """Translate the entire object by a vector offset.

        Args:
            axis (Vector3): Translation vector to move the object
        """
        for face_idx, (face, material) in enumerate(self.faces):
            moved_face = [vertex + axis for vertex in face]
            self.faces[face_idx] = (moved_face, material)

        self._invalidate_cache()

    def get_face_data(self, face_idx):
        """Get cached or computed face centroid and normal data.

        Args:
            face_idx (int): Index of the face to get data for

        Returns:
            tuple: (centroid, normal) where centroid is Vector3 center point
                and normal is Vector3 normalized face normal
        """
        if not self.use_caching:
            return self._compute_face_data(face_idx)

        cache_key = (face_idx, self._object_version)
        cached = self._face_cache.get(cache_key)
        if cached is not None:
            return cached

        result = self._compute_face_data(face_idx)
        self._face_cache[cache_key] = result
        return result

    def _compute_face_data(self, face_idx):
        """Compute face centroid and normal without caching.

        Args:
            face_idx (int): Index of the face to compute data for

        Returns:
            tuple: (centroid, normal) where centroid is Vector3 center point
                and normal is Vector3 normalized face normal
        """
        face, _ = self.faces[face_idx]
        centroid = calculate_face_centroid(face)
        normal = calculate_face_normal(face)
        return (centroid, normal)


def load_obj(file_path, scale=0) -> List[Object]:
    """Load 3D objects from Wavefront OBJ file format with improved error handling.

    Args:
        file_path (str): Path to the .obj file to load
        scale (float, optional): Scaling factor for all vertices. Defaults to 0 (no scaling)

    Returns:
        List[Object]: List of Object instances loaded from the file

    Raises:
        FileNotFoundError: If the specified file path doesn't exist
        ValueError: If the file format is invalid or corrupted
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"OBJ file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading OBJ file {file_path}: {str(e)}")

    objects = []
    vertexes = []
    curr_mtl = None

    # Ensure we have at least one default object
    default_object_created = False

    for line_num, line in enumerate(lines, 1):
        try:
            parts = line.strip().split()
            if not parts or parts[0].startswith("#"):  # Skip empty lines and comments
                continue
            command, *args = parts

            if command == "o" and args:
                name = args[0]
                objects.append(Object(name, []))
                default_object_created = True
            elif command == "v" and len(args) >= 3:
                if scale == 0:
                    vertexes.append(Vector3(*map(float, args[:3])))
                else:
                    vertexes.append(Vector3(*map(lambda x: float(x) * scale, args[:3])))
            elif command == "f" and args:
                # Create default object if none exists
                if not objects:
                    objects.append(Object("default", []))
                    default_object_created = True

                # Parse face indices, handling different formats
                face_vertices = []
                for vertex_data in args:
                    try:
                        # Handle format "v/vt/vn" or just "v"
                        vertex_index = int(vertex_data.split("/")[0])
                        # Handle negative indices (relative to current vertex count)
                        if vertex_index < 0:
                            vertex_index = len(vertexes) + vertex_index + 1
                        face_vertices.append(vertexes[vertex_index - 1])
                    except (ValueError, IndexError):
                        continue

                if len(face_vertices) >= 3:  # Only add valid faces with 3+ vertices
                    objects[-1].faces.append((face_vertices, curr_mtl))
            elif command == "usemtl" and args:
                curr_mtl = args[0]
        except Exception as e:
            print(f"Warning: Error parsing line {line_num} in {file_path}: {str(e)}")
            continue

    if not objects:
        objects.append(Object("default", []))

    return objects


def load_mtl(file_path, objects) -> List[Material]:
    """Load materials from Wavefront MTL file format with improved error handling.

    Args:
        file_path (str): Path to the .mtl file to load
        objects (Object or List[Object]): Objects to associate materials with

    Returns:
        List[Material]: List of Material instances with object-prefixed names

    Raises:
        FileNotFoundError: If the specified file path doesn't exist
    """
    if isinstance(objects, Object):
        objects = [objects]

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Warning: MTL file not found: {file_path}")
        return []
    except Exception as e:
        print(f"Warning: Error reading MTL file {file_path}: {str(e)}")
        return []

    materials = []
    curr_mtl = None
    curr_kd = [1.0, 1.0, 1.0]
    curr_ks = [0.0, 0.0, 0.0]
    curr_ns = 0.0

    for line_num, line in enumerate(lines, 1):
        try:
            parts = line.strip().split()
            if not parts or parts[0].startswith("#"):  # Skip empty lines and comments
                continue
            command, *args = parts

            if command == "newmtl" and args:
                if curr_mtl:
                    for obj in objects:
                        materials.append(
                            Material(
                                f"{obj.name}.{curr_mtl}", curr_kd, curr_ks, curr_ns
                            )
                        )
                curr_mtl = args[0]
                curr_kd = [1.0, 1.0, 1.0]
                curr_ks = [0.0, 0.0, 0.0]
                curr_ns = 0.0
            elif command == "Kd" and curr_mtl and len(args) >= 3:
                curr_kd = [max(0.0, min(1.0, float(i))) for i in args[:3]]
            elif command == "Ks" and curr_mtl and len(args) >= 3:
                curr_ks = [max(0.0, min(1.0, float(i))) for i in args[:3]]
            elif command == "Ns" and curr_mtl and args:
                curr_ns = max(0.0, min(1000.0, float(args[0])))
        except Exception as e:
            print(f"Warning: Error parsing line {line_num} in {file_path}: {str(e)}")
            continue

    if curr_mtl:
        for obj in objects:
            materials.append(
                Material(f"{obj.name}.{curr_mtl}", curr_kd, curr_ks, curr_ns)
            )

    return materials


def resource_path(relative_path):
    """Get absolute path to resource file, works for dev and PyInstaller bundle.

    This function handles path resolution for both development environments and
    PyInstaller-bundled executables by checking for the _MEIPASS attribute.

    Args:
        relative_path (str): Relative path to the resource file

    Returns:
        str: Absolute path to the resource file
    """
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)


# main
if __name__ == "__main__":
    turtle.bgcolor("#181818")
    use_caching = True

    lighting = LightingConfig(
        use_caching=use_caching,
        enable_shadows=True,
        use_advanced_lighting=True,
        light_bounces=200,
        light_bounce_samples=8,
        precompute_bounces=True,
        max_light_distance=25.0,
        max_bounce_distance=128,
        shadow_bias=0.001,
        light_contribution_threshold=0.001,
    )

    lighting.add_light_source(
        AmbientLight(
            color="#FFFFFF",
            brightness=0.15,
        )
    )

    lighting.add_light_source(
        PointLight(
            pos=Vector3(-0.5, 1, -1.5),
            color="#008cff",
            brightness=1.0,
            falloff_type="linear",
            falloff_rate=0.06,
        )
    )
    lighting.add_light_source(
        PointLight(
            pos=Vector3(0.5, 1, -1.5),
            color="#04ff00",
            brightness=1.0,
            falloff_type="linear",
            falloff_rate=0.06,
        )
    )
    lighting.add_light_source(
        PointLight(
            pos=Vector3(-0.5, -1, -1.5),
            color="#ffb300",
            brightness=1.0,
            falloff_type="linear",
            falloff_rate=0.06,
        )
    )
    lighting.add_light_source(
        PointLight(
            pos=Vector3(0.5, -1, -1.5),
            color="#f2ff00",
            brightness=1.0,
            falloff_type="linear",
            falloff_rate=0.06,
        )
    )
    lighting.add_light_source(
        PointLight(
            pos=Vector3(0.5, 0, 2),
            color="#ff0000",
            brightness=1.0,
            falloff_type="linear",
            falloff_rate=0.06,
        )
    )
    lighting.add_light_source(
        PointLight(
            pos=Vector3(-0.5, 0, 2),
            color="#aa00ff",
            brightness=1.0,
            falloff_type="linear",
            falloff_rate=0.06,
        )
    )

    camera = Camera(
        Vector3(3, 1, -2),
        Vector3(-1, -0.5, 0),
        lighting,
        use_caching=use_caching,
    )

    cube = load_obj(resource_path("objs/cube.obj"), scale=0.2)
    cube_mtl = load_mtl(resource_path("objs/cube.mtl"), cube)
    cube[0].move(Vector3(0, 0, -4))

    blahaj = load_obj(resource_path("objs/blahaj.obj"), scale=0.5)
    blahaj_mtl = load_mtl(resource_path("objs/blahaj.mtl"), blahaj)

    objects = cube + blahaj

    lighting.set_precompute_objects(objects)
    lighting.precompute_direct_lighting(objects)
    lighting.precompute_bounce_lighting(objects)

    for obj in objects:
        obj.precompute_face_cache()

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
        momentum *= 0.85

        camera.render(
            cube + blahaj,
            cube_mtl + blahaj_mtl,
            fov=103,
        )

        turtle.update()

        frame_count += 1
        if frame_count % 120 == 0:
            stats = lighting.get_cache_stats()
            print(
                f"Frame {frame_count} | Cache hit rate: {stats['hit_rate']:.1f}% | "
                f"Lighting cache: {stats['lighting_cache_size']} | "
                f"Face cache: {stats['face_lighting_cache_size']} | "
                f"Bounce cache: {stats['bounce_cache_size']} (valid: {stats['bounce_cache_valid']})"
            )

        if keyboard.is_pressed("p"):
            mouse_locked = not mouse_locked
            time.sleep(0.2)

        if keyboard.is_pressed("esc"):
            mouse_locked = False

    while True:
        main()
