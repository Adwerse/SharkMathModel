// Допоміжні функції (як ти дав)
function kmToLatDegrees(km) {
  const kmPerDegreeLat = 111.32; // середнє км в 1° широти
  return km / kmPerDegreeLat;
}

function kmToLngDegrees(km, latitude) {
  const kmPerDegreeLng = 111.32 * Math.cos(latitude * Math.PI / 180);
  // Якщо cos~0 (на полюсах) — повертаємо дуже мале значення, щоб не зациклитись
  if (Math.abs(kmPerDegreeLng) < 1e-9) return 0;
  return km / kmPerDegreeLng;
}

/**
 * Генерує точки всередині bbox кроком km (в км).
 * Повертає масив { position: [lng, lat], value: <рандом 0..100> }.
 *
 * @param {{minLng:number,minLat:number,maxLng:number,maxLat:number}} bbox
 * @param {number} km
 * @returns {Array<{position:[number,number], value:number}>}
 */
function generatePoints(bbox, km) {
  if (!bbox || typeof km !== 'number' || km <= 0) {
    throw new Error('Невірні аргументи: передай bbox і додатнє число km');
  }

  const { minLng, minLat, maxLng, maxLat } = bbox;
  if (minLat > maxLat || minLng > maxLng) {
    throw new Error('bbox некоректний (min > max)');
  }

  const latStep = kmToLatDegrees(km);
  if (latStep <= 0) throw new Error('latStep вийшов <= 0');

  const points = [];

  // Кількість рядів/стовпчиків — беремо ceil, щоб покрити весь діапазон
  const rowCount = Math.ceil((maxLat - minLat) / latStep);

  for (let i = 0; i <= rowCount; i++) {
    // Вираховуємо широту для цього ряду, останній ряд точно ставимо на maxLat
    let lat = minLat + i * latStep;
    if (lat > maxLat) lat = maxLat;

    // Крок довготи в градусах для поточної широти
    const lngStep = kmToLngDegrees(km, lat);
    if (lngStep <= 0) continue; // на випадок полюса або проблеми — пропускаємо

    const colCount = Math.ceil((maxLng - minLng) / lngStep);

    for (let j = 0; j <= colCount; j++) {
      let lng = minLng + j * lngStep;
      if (lng > maxLng) lng = maxLng;

      points.push({
        position: [lng, lat],            // [lng, lat] — як в deck.gl
        value: Math.round(Math.random() * 100) // випадкове значення 0..100
      });
    }
  }

  return points;
}