"""Testes do modelo cinemático de bicicleta (env/car_model.py)."""
import numpy as np

from env.car_model import KinematicBicycleModel, VehicleParams


def _car(v=0.0, dt=0.02):
    car = KinematicBicycleModel(VehicleParams(), dt=dt)
    car.reset(v=v)
    return car


def test_friction_circle_bounds_total_accel():
    """sqrt(ax^2 + ay^2) nunca deve exceder mu*g (friction circle)."""
    car = _car(v=20.0)
    mu_g = car.params.mu * 9.81
    for _ in range(100):
        car.step(np.array([1.0, 1.0]))  # steering e throttle no maximo
        total = float(np.hypot(car.ax, car.ay))
        assert total <= mu_g * 1.01, f"friction circle violado: {total:.2f} > {mu_g:.2f}"


def test_steering_rate_limited():
    """Um salto de comando de steering respeita max_steering_rate * dt."""
    car = _car(v=10.0)
    max_delta = car.params.max_steering_rate * car.dt
    prev = car.steering
    car.step(np.array([1.0, 0.0]))  # pedir steering maximo de repente
    assert abs(car.steering - prev) <= max_delta + 1e-9


def test_steering_clamped_to_max():
    """O angulo de steering nunca ultrapassa max_steering."""
    car = _car(v=5.0)
    for _ in range(200):
        car.step(np.array([1.0, 0.0]))
    assert abs(car.steering) <= car.params.max_steering + 1e-9


def test_speed_capped_at_max():
    """Acelerar continuamente nao ultrapassa max_speed."""
    car = _car(v=0.0)
    for _ in range(1000):
        car.step(np.array([0.0, 1.0]))
    assert car.v <= car.params.max_speed + 1e-6


def test_speed_non_negative_on_braking():
    """Travar continuamente nao produz velocidade negativa (sem marcha-atras)."""
    car = _car(v=10.0)
    for _ in range(1000):
        car.step(np.array([0.0, -1.0]))
    assert car.v >= 0.0


def test_braking_decelerates():
    car = _car(v=15.0)
    car.step(np.array([0.0, -1.0]))
    assert car.v < 15.0


def test_corners_shape():
    """Os 4 cantos da bounding box do carro tem forma (4, 2)."""
    car = _car()
    corners = car.get_corners()
    assert corners.shape == (4, 2)
