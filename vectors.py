"""
Vector2 and Vector3
"""

import math


class Vector2:
    def __init__(self, x: int, y: int):
        self.x, self.y = x, y

    def __repr__(self):
        return f"Vector2({self.x}, {self.y})"

    def __iter__(self):
        yield self.x
        yield self.y

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float):
        return Vector2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float):
        return Vector2(self.x / scalar, self.y / scalar)

    def __floordiv__(self, scalar: int):
        return Vector2(self.x // scalar, self.y // scalar)

    def __neg__(self):
        return Vector2(-self.x, -self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def magnitude(self):
        return math.sqrt((self.x * self.x + self.y * self.y))

    def normalize(self):
        mag = self.magnitude()
        return Vector2(self.x / mag, self.y / mag) if mag != 0 else Vector2(0, 0)


class Vector3:
    def __init__(self, x: float, y: float, z: float):
        self.x, self.y, self.z = x, y, z

    def __repr__(self):
        return f"Vector3({self.x}, {self.y}, {self.z})"

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float):
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __floordiv__(self, scalar: int):
        return Vector3(self.x // scalar, self.y // scalar, self.z // scalar)

    def __neg__(self):
        return Vector3(-self.x, -self.y, -self.z)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def magnitude(self):
        return math.sqrt((self.x * self.x + self.y * self.y + self.z * self.z))

    def normalize(self):
        mag = self.magnitude()
        return (
            Vector3(self.x / mag, self.y / mag, self.z / mag)
            if mag != 0
            else Vector3(0, 0, 0)
        )
