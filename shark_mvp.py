from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Iterable

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse


@dataclass(frozen=True)
class Hotspot:
    name: str
    lat: float
    lon: float
    weight: float
    radius_km: float


HOTSPOTS: tuple[Hotspot, ...] = (
    Hotspot("Eastern Australia", -25.5, 153.0, 0.22, 750.0),
    Hotspot("South Africa", -34.0, 18.5, 0.18, 550.0),
    Hotspot("Hawaii", 20.9, -157.0, 0.14, 450.0),
    Hotspot("Florida", 27.5, -80.0, 0.16, 500.0),
    Hotspot("California", 34.0, -119.5, 0.12, 500.0),
    Hotspot("Brazil", -22.9, -43.2, 0.10, 420.0),
    Hotspot("Red Sea", 20.0, 38.5, 0.10, 350.0),
)


@dataclass(frozen=True)
class LandRegion:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    penalty: float


# Грубые рамки для континентальных внутренних областей (подавляют вероятность на суше).
LAND_INTERIORS: tuple[LandRegion, ...] = (
    LandRegion(25.0, 50.0, -110.0, -90.0, 0.35),   # Центральные штаты США
    LandRegion(-20.0, 5.0, -70.0, -50.0, 0.40),    # Внутренняя часть бассейна Амазонки
    LandRegion(-10.0, 20.0, 10.0, 35.0, 0.45),     # Центральная Африка
    LandRegion(30.0, 60.0, 40.0, 120.0, 0.50),     # Евразийский континентальный массив
    LandRegion(-35.0, -15.0, 120.0, 145.0, 0.65),  # Центральная и западная Австралия (сильный штраф)
)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _wrap_longitude(lon: float) -> float:
    """Привести долготу к диапазону [-180, 180)."""
    wrapped = (lon + 180.0) % 360.0 - 180.0
    # Обработать частный случай, когда долгота ровно 180.
    return wrapped if wrapped != -180.0 else 180.0


def _haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Приблизительное расстояние по дуге между двумя точками на Земле."""
    radius_earth_km = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_earth_km * c


def _lat_factor(lat: float) -> float:
    """Отдаёт предпочтение широтам в тропиках и тёплых прибрежных водах."""
    # Квадрат косинуса плавно уменьшается от экватора к полюсам.
    equatorial_bias = math.cos(math.radians(lat)) ** 2
    # Дополнительное усиление внутри тропиков (|lat| <= 30°).
    tropic_span = clamp(1.0 - abs(lat) / 30.0)
    # Небольшое штрафование за экстремальные широты у полюсов.
    polar_penalty = clamp((abs(lat) - 50.0) / 40.0)
    return clamp(0.6 * equatorial_bias + 0.4 * tropic_span - 0.3 * polar_penalty, 0.0, 1.0)


def _current_factor(lat: float, lon: float) -> float:
    """Эвристика для тёплых течений и продуктивных вод."""
    lon_rad = math.radians(_wrap_longitude(lon))
    lat_rad = math.radians(lat)
    # Комбинируем синусоидальные паттерны, чтобы приблизить известные тёплые течения.
    warm_current = 0.5 * (1.0 + math.sin(lon_rad * 1.5) * math.cos(lat_rad))
    upwelling = 0.5 * (1.0 + math.cos(lon_rad * 0.7 + math.sin(lat_rad)))
    return clamp(0.6 * warm_current + 0.4 * upwelling, 0.0, 1.0)


def _hotspot_contribution(lat: float, lon: float, hotspots: Iterable[Hotspot]) -> float:
    score = 0.0
    for spot in hotspots:
        distance = _haversine_distance_km(lat, lon, spot.lat, spot.lon)
        influence = math.exp(-(distance / spot.radius_km) ** 2)
        score += spot.weight * influence
    return score


def _land_penalty(lat: float, lon: float, regions: Iterable[LandRegion]) -> float:
    penalty = 0.0
    for region in regions:
        if region.lat_min <= lat <= region.lat_max and region.lon_min <= lon <= region.lon_max:
            penalty += region.penalty
    return penalty


def shark_probability(lat: float, lon: float) -> float:
    """Оценить вероятность встречи акул в заданных координатах.

    Функция отдаёт приоритет тропическим и тёплым умеренным водам,
    подчёркивает глобальные очаги активности и снижает вероятность для
    широких континентальных областей.

    Аргументы:
        lat: Широта в десятичных градусах (от -90 до 90).
        lon: Долгота в десятичных градусах (от -180 до 180).

    Возвращает:
        Вероятность в диапазоне от 0 до 1, округлённую до трёх знаков.
    """
    if not math.isfinite(lat) or not math.isfinite(lon):
        raise ValueError("Latitude and longitude must be finite numbers.")

    lat = clamp(lat, -90.0, 90.0)
    lon = _wrap_longitude(lon)

    base = 0.05
    lat_component = 0.35 * _lat_factor(lat)
    current_component = 0.20 * _current_factor(lat, lon)
    hotspot_component = _hotspot_contribution(lat, lon, HOTSPOTS)
    land_penalty = _land_penalty(lat, lon, LAND_INTERIORS)

    probability = clamp(base + lat_component + current_component + hotspot_component - land_penalty)
    return round(probability, 3)


class SharkRequestHandler(BaseHTTPRequestHandler):
    """Обработчик HTTP-запросов, возвращающий вероятность в формате JSON."""

    def do_GET(self) -> None:  # noqa: N802 (сигнатура фиксирована HTTPServer)
        parsed = urlparse(self.path)
        if parsed.path != "/probability":
            self._send_json(404, {"error": "not_found"})
            return

        params = parse_qs(parsed.query)
        try:
            lat = float(params.get("lat", [])[0])
            lon = float(params.get("lon", [])[0])
        except (IndexError, ValueError):
            self._send_json(400, {"error": "lat and lon query params are required floats"})
            return

        try:
            cleaned_lat = clamp(lat, -90.0, 90.0)
            cleaned_lon = _wrap_longitude(lon)
            probability = shark_probability(lat, lon)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        self._send_json(200, {"lat": cleaned_lat, "lon": cleaned_lon, "probability": probability})

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003 (совместимость с базовым классом)
        """Подавляем стандартный вывод сервера, чтобы не шуметь в консоли."""

        return

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Запустить HTTP-сервер для получения вероятности по GET-запросам."""

    server = HTTPServer((host, port), SharkRequestHandler)
    print(f"Shark probability server listening on http://{host}:{port}")
    print("Send GET /probability?lat=<value>&lon=<value>")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


def run_cli() -> None:
    """Интерактивный ввод координат через консоль."""

    print("Shark presence probability estimator")
    try:
        user_lat = float(input("Enter latitude (-90 to 90): ").strip())
        user_lon = float(input("Enter longitude (-180 to 180): ").strip())
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    try:
        probability = shark_probability(user_lat, user_lon)
    except ValueError as exc:
        print(f"Error: {exc}")
    else:
        print(f"Estimated probability of encountering sharks: {probability:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Эвристический расчёт вероятности встречи акул.")
    parser.add_argument("--server", action="store_true", help="Запустить HTTP-сервер для фронтенда.")
    parser.add_argument("--host", default="0.0.0.0", help="Хост, на котором слушает сервер (по умолчанию 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8000, help="Порт сервера (по умолчанию 8000).")
    args = parser.parse_args()

    if args.server:
        run_server(args.host, args.port)
    else:
        run_cli()


if __name__ == "__main__":
    main()
