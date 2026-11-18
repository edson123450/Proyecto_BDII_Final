const $ = (id) => document.getElementById(id);
const escapeHTML = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const examples = {
  range:
    "SELECT id, nombre, dept, salario\nFROM empleados\nWHERE id >= 100 AND id <= 150;",
  equal: "SELECT * FROM empleados WHERE id = 42000;",
  hash: "SELECT id, nombre, dept FROM empleados\nWHERE dept = 'Sistemas' LIMIT 100;",
  scan: "SELECT * FROM empleados_seq\nWHERE salario >= 13500 LIMIT 100;",
  spatial: "SELECT id, nombre, ubicacion FROM locales\nWHERE ubicacion WITHIN_RADIUS((-77.04,-12.05),2.5);",
  text: "SELECT id, titulo, COSINE_SCORE() AS relevancia\nFROM documentos WHERE cuerpo MATCH 'jazz AND guitar'\nORDER BY relevancia DESC LIMIT 10;",
  create:
    "CREATE TABLE prueba (id INT PRIMARY KEY, nombre CHAR(30), salario FLOAT) USING HEAP;",
  insert: "INSERT INTO prueba VALUES (1, 'Ada Lovelace', 5200.0);",
};
let page = 1,
  total = 0,
  lastSQL = "";
function highlight() {
  const source = $("sql").value;
  const tokens = source.split(
    /('(?:''|[^'])*'|\b(?:SELECT|FROM|WHERE|AND|OR|INSERT|INTO|VALUES|CREATE|TABLE|INDEX|ON|USING|HEAP|SEQUENTIAL|BTREE|HASH|DELETE|PRIMARY|KEY|INT|CHAR|FLOAT|LIMIT|ORDER|BY|ASC|DESC|MATCH|RTREE|INVERTED|IVF|WITH|AS|SET|COSINE_SCORE|WITHIN_RADIUS|WITHIN_WINDOW|KNN)\b|\b\d+(?:\.\d+)?\b)/gi,
  );
  $("highlight").innerHTML =
    tokens
      .map((t) =>
        /^'/.test(t)
          ? `<span class="string">${escapeHTML(t)}</span>`
          : /^\d/.test(t)
            ? `<span class="number">${escapeHTML(t)}</span>`
            : /^[A-Z]+$/i.test(t) &&
                /^(SELECT|FROM|WHERE|AND|OR|INSERT|INTO|VALUES|CREATE|TABLE|INDEX|ON|USING|HEAP|SEQUENTIAL|BTREE|HASH|DELETE|PRIMARY|KEY|INT|CHAR|FLOAT|LIMIT|ORDER|BY|ASC|DESC|MATCH|RTREE|INVERTED|IVF|WITH|AS|SET|COSINE_SCORE|WITHIN_RADIUS|WITHIN_WINDOW|KNN)$/i.test(
                  t,
                )
              ? `<span class="kw">${escapeHTML(t)}</span>`
              : escapeHTML(t),
      )
      .join("") + "\n";
}
async function request(url, options = {}) {
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); }
  catch { throw new Error(`El servidor no devolvió una respuesta válida (${response.status}).`); }
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : data.detail ? JSON.stringify(data.detail) : `No se pudo completar la solicitud (${response.status}).`);
  return data;
}
async function loadTables() {
  try {
    const tables = await request("/api/tables");
    $("tables").innerHTML = tables.length
      ? tables
          .map(
            (t) =>
              `<div class="table-card"><button class="table-name" data-table="${escapeHTML(t.name)}">▤ ${escapeHTML(t.name)}</button><div class="table-sub"><span class="chip">${escapeHTML(t.organization)}</span><span>${Number(t.record_count).toLocaleString("es-PE")} filas</span></div>${(t.columns || []).map((c) => `<div class="column"><span>${c.primary ? "⌑ " : ""}${escapeHTML(c.name)}</span><code>${escapeHTML(c.type)}${c.size ? "(" + c.size + ")" : ""}</code></div>`).join("")}${(t.indexes || []).map((i) => `<div class="index-item">↳ ${escapeHTML(i.name)} · ${escapeHTML(i.kind)}</div>`).join("")}${t.organization === "SEQUENTIAL" ? `<button class="secondary reorganize" data-reorganize="${escapeHTML(t.name)}">Reorganizar</button>` : ""}</div>`,
          )
          .join("")
      : '<p class="empty">Crear una tabla para comenzar.</p>';
    $("tables")
      .querySelectorAll("[data-table]")
      .forEach(
        (b) =>
          (b.onclick = () => {
            $("sql").value = `SELECT * FROM ${b.dataset.table} LIMIT 100;`;
            highlight();
          }),
      );
    $("tables")
      .querySelectorAll("[data-reorganize]")
      .forEach(
        (b) =>
          (b.onclick = async () => {
            b.disabled = true;
            try {
              const d = await request("/api/tables/reorganize", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ table: b.dataset.reorganize }),
              });
              $("message").textContent = d.message;
              renderMetrics(d.metrics, {
                operation: "Reorganize",
                reason:
                  "Fusión del área principal con overflow y reconstrucción de índices.",
              });
              await loadTables();
            } catch (e) {
              showError(e);
            } finally {
              b.disabled = false;
            }
          }),
      );
  } catch (e) {
    showError(e);
  }
}
function showError(error) {
  $("message").textContent = error.message;
  $("message").className = "status-message error";
}
function renderMetrics(m, p) {
  $("reads").textContent = Number(m.disk_reads || 0).toLocaleString("es-PE");
  $("writes").textContent = Number(m.disk_writes || 0).toLocaleString("es-PE");
  $("execution").textContent = Number(m.execution_ms || 0).toFixed(3) + " ms";
  $("parse").textContent = Number(m.parse_ms || 0).toFixed(3) + " ms";
  $("plan-name").textContent = p.operation || "Operación completada";
  $("plan-reason").textContent =
    p.reason || "Operación completada sobre los archivos del motor.";
  $("plan-index").textContent = p.index || "";
  $("plan-strategy").hidden = !p.strategy;
  $("plan-strategy").textContent = p.strategy ? p.strategy + (p.filter?.index ? ' (' + p.filter.index + ')' : '') : '';
  $("plan-details").hidden = false;
  $("plan-json").textContent = JSON.stringify(p, null, 2);
  const max = Math.max(m.disk_reads || 0, m.disk_writes || 0, 1);
  $("read-bar").style.width = ((m.disk_reads || 0) / max) * 100 + "%";
  $("write-bar").style.width = ((m.disk_writes || 0) / max) * 100 + "%";
  const roles = m.by_role || {};
  $("io-detail").innerHTML = Object.entries(roles)
    .map(
      ([name, v]) =>
        `${escapeHTML(name === "index" ? "Índices" : "Datos")}: ${v.disk_reads || 0} lecturas · ${v.disk_writes || 0} escrituras`,
    )
    .join("<br>");
}
async function execute(reset = true) {
  if (reset) {
    page = 1;
    lastSQL = $("sql").value;
  }
  const button = $("execute");
  button.disabled = true;
  button.textContent = "Ejecutando…";
  $("message").className = "status-message";
  try {
    const result = await request("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sql: lastSQL, page, page_size: 25 }),
    });
    total = Number(result.total || 0);
    $("message").textContent = result.message;
    $("result-count").textContent =
      total.toLocaleString("es-PE") + " registros";
    if (result.rows.length) {
      const columns = Object.keys(result.rows[0]);
      $("results").innerHTML =
        `<table><thead><tr>${columns.map((c) => `<th>${escapeHTML(c)}</th>`).join("")}</tr></thead><tbody>${result.rows.map((row) => `<tr>${columns.map((c) => `<td>${escapeHTML(row[c])}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
    } else {
      $("results").innerHTML =
        '<div class="empty"><span>✓</span><p>' +
        escapeHTML(result.message) +
        "</p></div>";
    }
    renderMetrics(result.metrics, result.plan);
    $("page-label").textContent =
      `Página ${page} de ${Math.max(1, Math.ceil(total / 25))}`;
    $("previous").disabled = page <= 1;
    $("next").disabled = page * 25 >= total;
    await loadTables();
  } catch (e) {
    showError(e);
  } finally {
    button.disabled = false;
    button.textContent = "▶ Ejecutar consulta";
  }
}
$("sql").addEventListener("input", highlight);
$("sql").addEventListener("scroll", () => {
  $("highlight").scrollTop = $("sql").scrollTop;
});
$("sql").addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    execute();
  }
});
$("examples").onchange = () => {
  $("sql").value = examples[$("examples").value];
  highlight();
};
$("execute").onclick = () => execute();
$("refresh").onclick = loadTables;
$("previous").onclick = () => {
  page--;
  execute(false);
};
$("next").onclick = () => {
  page++;
  execute(false);
};
highlight();
loadTables();
request("/api/health")
  .then(() => {
    $("connection").textContent = "Motor conectado";
  })
  .catch(() => {
    $("connection").textContent = "Sin conexión";
  });

// Las respuestas se presentan como texto; los medios solo usan rutas del servidor.
function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}
function number(value, decimals = 0) {
  return Number.isFinite(Number(value)) ? Number(value).toLocaleString("es-PE", { maximumFractionDigits: decimals, minimumFractionDigits: decimals }) : "—";
}
function localURL(value, prefix) {
  if (!value) return null;
  try {
    const url = new URL(value, location.origin);
    return url.origin === location.origin && url.pathname.startsWith(prefix) ? url.href : null;
  } catch { return null; }
}
function status(id, message, state = "") {
  $(id).textContent = message;
  $(id).className = `status-message${state ? " " + state : ""}`;
}
function postJSON(url, body) {
  return request(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}
function rolesView(metrics) {
  const wrapper = element("div", "metric-roles");
  Object.entries(metrics.by_role || {}).forEach(([role, counts]) => {
    const row = element("div", "metric-role");
    row.append(element("span", "", role === "index" ? "Índices" : role === "data" ? "Datos" : role));
    row.append(element("strong", "", `${number(counts.disk_reads)} lect. · ${number(counts.disk_writes)} escr.`));
    wrapper.append(row);
  });
  return wrapper;
}
function metricPanel(id, metrics = {}, plan = {}) {
  const container = $(id);
  const title = element("div", "panel-title");
  const heading = element("h2", "", "Plan"); heading.append(element("span", "", "y métricas")); title.append(heading);
  const block = element("div", "plan-block");
  block.append(element("span", "eyebrow", "RUTA DE ACCESO"), element("h3", "", plan.operation || "Por consultar"));
  block.append(element("p", "", plan.reason || "El motor mostrará el recorrido y los accesos de la consulta."));
  if (plan.index) block.append(element("code", "", plan.index));
  const values = element("div", "mini-metrics");
  [["Lecturas", metrics.disk_reads, 0, ""], ["Escrituras", metrics.disk_writes, 0, ""], ["Ejecución", metrics.execution_ms, 3, " ms"], ["Parseo", metrics.parse_ms, 3, " ms"]].forEach(([name, value, precision, unit]) => {
    const item = element("div"); item.append(element("span", "", name), element("strong", unit ? "time-value" : "", value === undefined ? "—" : number(value, precision) + unit)); values.append(item);
  });
  container.replaceChildren(title, block, values, rolesView(metrics));
  if (Object.keys(plan).length) {
    const details = element("details", "plan-details");
    details.append(element("summary", "", "Detalles del plan"), element("pre", "", JSON.stringify(plan, null, 2)));
    container.append(details);
  }
  container.append(element("p", "metric-note", "I/O cuenta páginas solicitadas por el gestor. El sistema operativo puede conservarlas en su caché."));
}
function dataTable(container, rows, emptyText = "No se encontraron resultados.") {
  if (!rows.length) {
    const empty = element("div", "empty"); empty.append(element("p", "", emptyText)); container.replaceChildren(empty); return;
  }
  const table = element("table"), head = element("thead"), headRow = element("tr"), body = element("tbody");
  const columns = Object.keys(rows[0]);
  columns.forEach((name) => headRow.append(element("th", "", name))); head.append(headRow);
  rows.forEach((row) => {
    const line = element("tr");
    columns.forEach((name) => line.append(element("td", "", typeof row[name] === "object" && row[name] !== null ? JSON.stringify(row[name]) : row[name] ?? "")));
    body.append(line);
  });
  table.append(head, body); container.replaceChildren(table);
}
const views = {
  sql: ["Los datos, desde adentro.", "Ejecutar SQL y observar cómo trabaja el motor sobre sus páginas."],
  map: ["Encontrar lo que está cerca.", "Explora lugares de Lima por distancia, cercanía o área."],
  text: ["Las palabras también conectan.", "Busca documentos musicales y compara su relevancia."],
  audio: ["Reconocer una canción, por un fragmento.", "Escucha, recorta y compara coincidencias con distintos métodos."],
  experiments: ["Medir para entender.", "Resultados reales de las pruebas de almacenamiento, búsqueda y similitud."],
};
let activeView = "sql", audioLoaded = false, mapSQL = "", textSQL = "";
function switchView(name) {
  if (!views[name]) return;
  if (activeView === "audio" && name !== "audio") document.querySelectorAll("audio").forEach((player) => player.pause());
  activeView = name;
  Object.keys(views).forEach((view) => { $("view-" + view).hidden = view !== name; });
  document.querySelectorAll("[data-view]").forEach((button) => {
    const selected = button.dataset.view === name;
    button.classList.toggle("active", selected);
    if (selected) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current");
  });
  $("view-title").textContent = views[name][0]; $("view-description").textContent = views[name][1];
  $("show-benchmarks").hidden = name === "experiments";
  if (name === "map") { initializeMap(); if (spatialMap) requestAnimationFrame(() => spatialMap.invalidateSize()); }
  if (name === "audio" && !audioLoaded) loadAudio();
  if (name === "experiments") loadExperiments();
}
function showSQL(sql) { if (sql) { $("sql").value = sql; highlight(); switchView("sql"); $("sql").focus(); } }
document.querySelectorAll("[data-view]").forEach((button) => { button.onclick = () => switchView(button.dataset.view); });
$("show-benchmarks").onclick = () => switchView("experiments");
$("close-benchmarks").onclick = () => switchView("sql");
$("map-show-sql").onclick = () => showSQL(mapSQL);
$("text-show-sql").onclick = () => showSQL(textSQL);
metricPanel("map-metrics"); metricPanel("text-metrics");

let spatialMap = null, pointLayer = null, queryLayer = null, tileWarning = false;
function mapParameters() {
  return { lon: Number($("map-lon").value), lat: Number($("map-lat").value), mode: $("map-mode").value, radius: $("map-mode").value === "radius" ? Number($("map-radius").value) : 2.5, k: Number($("map-k").value), width: $("map-mode").value === "window" ? Number($("map-width").value) : 5 };
}
function drawMapQuery() {
  if (!spatialMap) return;
  const input = mapParameters();
  if (!Number.isFinite(input.lon) || !Number.isFinite(input.lat)) return;
  queryLayer.clearLayers();
  L.circleMarker([input.lat, input.lon], { radius: 6, color: "#153737", weight: 2, fillColor: "#fff", fillOpacity: 1 }).addTo(queryLayer);
  const options = { color: "#c67328", weight: 2, dashArray: "7 5", fillColor: "#087f72", fillOpacity: .04 };
  if (input.mode === "radius" && input.radius > 0) L.circle([input.lat, input.lon], { ...options, radius: input.radius * 1000 }).addTo(queryLayer);
  if (input.mode === "window" && input.width > 0) {
    const dy = input.width / (2 * 111.32), dx = dy / Math.cos(-12.05 * Math.PI / 180);
    L.rectangle([[input.lat - dy, input.lon - dx], [input.lat + dy, input.lon + dx]], options).addTo(queryLayer);
  }
}
function initializeMap() {
  if (spatialMap) return;
  if (!window.L) { status("map-status", "No se pudo cargar el mapa. La búsqueda por coordenadas sigue disponible.", "error"); return; }
  spatialMap = L.map("spatial-map", { preferCanvas: true }).setView([-12.05, -77.04], 12);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>' })
    .on("tileerror", () => { if (!tileWarning) { tileWarning = true; status("map-status", "Mapa base sin conexión. Los puntos y las consultas siguen disponibles."); } }).addTo(spatialMap);
  pointLayer = L.layerGroup().addTo(spatialMap); queryLayer = L.layerGroup().addTo(spatialMap);
  spatialMap.on("click", (event) => { $("map-lon").value = event.latlng.lng.toFixed(5); $("map-lat").value = event.latlng.lat.toFixed(5); drawMapQuery(); });
  drawMapQuery();
}
function updateMapControls() {
  const mode = $("map-mode").value;
  $("radius-field").hidden = mode !== "radius"; $("map-k-field").hidden = mode !== "knn"; $("width-field").hidden = mode !== "window";
  $("map-radius").disabled = mode !== "radius"; $("map-k").disabled = mode !== "knn"; $("map-width").disabled = mode !== "window";
  drawMapQuery();
}
["map-mode", "map-lon", "map-lat", "map-radius", "map-width"].forEach((id) => $(id).addEventListener("change", updateMapControls));
$("map-form").addEventListener("submit", async (event) => {
  event.preventDefault(); const button = $("map-submit"); if (button.disabled) return;
  button.disabled = true; button.textContent = "Buscando…"; status("map-status", "Consultando los puntos del área…", "busy");
  try {
    const input = mapParameters(), result = await postJSON("/api/spatial/search", input), rows = result.rows || [];
    mapSQL = result.sql || ""; $("map-show-sql").disabled = !mapSQL;
    dataTable($("map-results"), rows); metricPanel("map-metrics", result.metrics, result.plan);
    const count = result.total ?? rows.length;
    $("map-count").textContent = `${number(rows.length)} lugares${count > rows.length ? " de " + number(count) : ""}`;
    status("map-status", result.message || `${number(count)} lugares encontrados.`);
    if (spatialMap) {
      pointLayer.clearLayers(); drawMapQuery(); const bounds = [];
      rows.forEach((row) => {
        if (!Array.isArray(row.ubicacion) || row.ubicacion.length !== 2 || !row.ubicacion.every(Number.isFinite)) return;
        const [lon, lat] = row.ubicacion, popup = element("div");
        popup.append(element("strong", "", row.nombre || `Lugar ${row.id}`), element("span", "", `ID ${row.id} · ${lat.toFixed(4)}, ${lon.toFixed(4)}`));
        L.circleMarker([lat, lon], { radius: rows.length > 1000 ? 2 : 4, color: "#087f72", weight: .5, fillColor: "#087f72", fillOpacity: .6 }).bindPopup(popup).addTo(pointLayer);
        bounds.push([lat, lon]);
      });
      drawMapQuery();
      bounds.push([input.lat, input.lon]);
      if (bounds.length > 1) spatialMap.fitBounds(bounds, { padding: [35, 35], maxZoom: 15 }); else spatialMap.setView([input.lat, input.lon], 13);
    }
  } catch (error) { status("map-status", error.message, "error"); }
  finally { button.disabled = false; button.textContent = "Buscar lugares"; }
});

function renderDocuments(rows) {
  const container = $("text-results"); container.replaceChildren();
  if (!rows.length) { const empty = element("div", "empty"); empty.append(element("p", "", "No se encontraron coincidencias. Prueba otras palabras.")); container.append(empty); return; }
  rows.forEach((row, index) => {
    const card = element("article", "document-card"), heading = element("div", "document-heading");
    heading.append(element("h3", "", row.titulo || row.title || `Documento ${row.id}`));
    const score = row.relevancia ?? row.score;
    if (score !== undefined) heading.append(element("span", "score-pill", `Relevancia ${number(score, 4)}`));
    const body = String(row.cuerpo || row.body || row.text || "");
    card.append(heading, element("p", "", body.length > 320 ? body.slice(0, 320) + "…" : body));
    if (body.length > 320) { const details = element("details"); details.append(element("summary", "", "Leer documento completo"), element("p", "", body)); card.append(details); }
    card.append(element("div", "document-meta", `#${index + 1} · Documento ${row.id}`)); container.append(card);
  });
}
$("text-form").addEventListener("submit", async (event) => {
  event.preventDefault(); const button = $("text-submit"); if (button.disabled) return;
  const query = $("text-query").value.trim(); if (!query) return;
  button.disabled = true; button.textContent = "Buscando…"; status("text-status", "Buscando y ordenando documentos…", "busy");
  try {
    const result = await postJSON("/api/text/search", { query, k: Number($("text-k").value) });
    renderDocuments(result.rows || []); metricPanel("text-metrics", result.metrics, result.plan);
    status("text-status", result.message || `${number((result.rows || []).length)} documentos encontrados.`);
    textSQL = result.sql || ""; $("text-show-sql").disabled = !textSQL;
  } catch (error) { status("text-status", error.message, "error"); }
  finally { button.disabled = false; button.textContent = "Buscar texto"; }
});
document.querySelectorAll("[data-text-query]").forEach((button) => { button.onclick = () => { $("text-query").value = button.dataset.textQuery; $("text-form").requestSubmit(); }; });

let audioReady = false, audioBusy = false, audioCatalog = [], uploadURL = null;
$('audio-show-sql').onclick = async () => {
  const id = $('audio-track').value;
  if (!id || $('audio-source').value !== 'catalog') return;
  try { const result = await request('/api/audio/sql/' + encodeURIComponent(id)); showSQL(result.sql); }
  catch (error) { status('audio-status', error.message, 'error'); }
};
async function loadAudio() {
  $("audio-refresh").disabled = true;
  const responses = await Promise.allSettled([request("/api/audio/info"), request("/api/audio/catalog?limit=100")]);
  const [infoResponse, catalogResponse] = responses;
  if (infoResponse.status === "fulfilled") {
    const info = infoResponse.value; audioReady = Boolean(info.ready); audioLoaded = true;
    $("audio-info").replaceChildren(element("strong", "", `${number(info.count || 0)} pistas`), element("span", "", audioReady ? "Colección lista para comparar fragmentos." : "La colección de audio aún se está preparando."));
    $("audio-corpus-count").textContent = `${number(info.count || 0)} pistas · ${number(info.dimension || 0)} dimensiones`;
  } else {
    audioReady = false; $("audio-info").textContent = infoResponse.reason.message;
  }
  if (catalogResponse.status === "fulfilled") {
    const payload = catalogResponse.value; audioCatalog = Array.isArray(payload) ? payload : payload.rows || payload.tracks || [];
    const previous = $("audio-track").value; $("audio-track").replaceChildren();
    if (!audioCatalog.length) $("audio-track").append(new Option("El catálogo todavía no está disponible", ""));
    audioCatalog.forEach((track) => $("audio-track").append(new Option(`${track.title || "Pista " + track.id} · ${track.artist || "Artista no indicado"}`, String(track.id))));
    if (audioCatalog.some((track) => String(track.id) === previous)) $("audio-track").value = previous;
    updateAudioSource();
  }
  $("audio-submit").disabled = !audioReady || audioBusy; $("audio-refresh").disabled = false;
}
function updateAudioSource() {
  const upload = $("audio-source").value === "upload";
  $("audio-show-sql").disabled = upload || !audioReady || !$("audio-track").value;
  $("track-field").hidden = upload; $("upload-field").hidden = !upload;
  const player = $("audio-source-player"); player.pause();
  if (uploadURL) { URL.revokeObjectURL(uploadURL); uploadURL = null; }
  let url = null;
  if (upload && $("audio-file").files[0]) { uploadURL = URL.createObjectURL($("audio-file").files[0]); url = uploadURL; }
  if (!upload && $("audio-track").value) url = `/api/audio/track/${encodeURIComponent($("audio-track").value)}`;
  player.hidden = !url;
  if (url) player.src = url; else { player.removeAttribute("src"); player.load(); }
  const track = audioCatalog.find((item) => String(item.id) === $("audio-track").value);
  const duration = Number(track?.duration);
  if (!upload && Number.isFinite(duration) && duration > 0) { const maximum = Math.max(0, duration - Number($("audio-duration").value)); $("audio-start").max = maximum; if (Number($("audio-start").value) > maximum) $("audio-start").value = maximum; } else $("audio-start").removeAttribute("max");
}
updateMapControls();
$("audio-source").onchange = updateAudioSource;
$("audio-track").onchange = () => { $("audio-start").value = "0"; updateAudioSource(); };
$("audio-file").onchange = updateAudioSource;
$("audio-duration").onchange = updateAudioSource;
$("audio-refresh").onclick = loadAudio;
document.addEventListener("play", (event) => { if (event.target.tagName === "AUDIO") document.querySelectorAll("audio").forEach((player) => { if (player !== event.target) player.pause(); }); }, true);
function methodLabel(name) {
  const value = String(name || "Método");
  if (/boaw/i.test(value)) return "Palabras acústicas · BoAW";
  if (/dtw/i.test(value)) return "Alineamiento temporal · DTW";
  if (/ivf/i.test(value)) return "Búsqueda aproximada · IVF";
  if (/exact|linear|sequential/i.test(value)) return "Vectores · Búsqueda exacta";
  return value;
}
function audioPlayer(url, label) {
  const player = element("audio"); player.controls = true; player.preload = "none"; player.setAttribute("aria-label", label); player.src = url; return player;
}
function renderAudioMethod(method) {
  const card = element("section", "panel method-card"), heading = element("div", "method-heading");
  const rows = method.rows || [], metrics = method.metrics || {}, plan = method.plan || {};
  heading.append(element("h3", "", methodLabel(method.name)), element("span", "badge", `${rows.length} resultados`));
  const brief = element("div", "method-brief");
  [[number(metrics.execution_ms ?? metrics.latency_ms ?? 0, 2) + " ms", "consulta"], [number(metrics.disk_reads || 0), "lecturas"], [number(metrics.disk_writes || 0), "escrituras"]].forEach(([value, label]) => { const item = element("span"); item.append(element("strong", "", value), document.createTextNode(" " + label)); brief.append(item); });
  card.append(heading, brief);
  const list = element("div", "track-results");
  rows.forEach((row, index) => {
    const item = element("article", "track-result"), details = element("div");
    item.append(element("span", "track-rank", String(index + 1).padStart(2, "0")));
    details.append(element("h4", "", row.title || `Pista ${row.id}`), element("p", "", [row.artist, row.genre].filter(Boolean).join(" · ")));
    const scores = element("div", "track-scores");
    if (row.score !== undefined) scores.append(element("span", "", `Similitud ${number(row.score, 4)}`));
    if (row.distance !== undefined) scores.append(element("span", "", `Distancia ${number(row.distance, 4)}`));
    if (row.start_seconds !== undefined) scores.append(element("span", "", `Desde ${number(row.start_seconds, 1)} s`));
    scores.append(element("span", "", `ID ${row.id}`)); details.append(scores);
    const url = localURL(row.audio_url, "/api/audio/"); if (url) details.append(audioPlayer(url, `Escuchar ${row.title || "pista " + row.id}`));
    item.append(details); list.append(item);
  });
  if (!rows.length) list.append(element("div", "empty", "Sin coincidencias para este método."));
  card.append(list);
  const details = element("details", "method-details");
  details.append(element("summary", "", "Ver plan y desglose de I/O"), rolesView(metrics));
  if (method.details?.windows !== undefined) {
    const stats = method.details;
    details.append(element("p", "", `${number(stats.tracks)} pistas · ${number(stats.windows)} ventanas · ${number(stats.pruned)} descartadas por LB_Keogh · ${number(stats.refined)} evaluadas con DTW.`));
  }
  details.append(element("pre", "", JSON.stringify(plan, null, 2))); card.append(details);
  return card;
}
$("audio-form").addEventListener("submit", async (event) => {
  event.preventDefault(); if (audioBusy || !audioReady) return;
  const upload = $("audio-source").value === "upload", file = $("audio-file").files[0];
  if (upload && !file) { status("audio-status", "Selecciona un archivo de audio para comparar.", "error"); return; }
  if (!upload && !$("audio-track").value) { status("audio-status", "Selecciona una pista del catálogo.", "error"); return; }
  const form = new FormData();
  if (upload) form.append("file", file); else form.append("track_id", $("audio-track").value);
  [["start", "audio-start"], ["duration", "audio-duration"], ["noise_db", "audio-noise"], ["k", "audio-k"], ["probes", "audio-probes"]].forEach(([name, id]) => form.append(name, $(id).value));
  form.append("include_dtw", $("audio-dtw").checked ? "true" : "false");
  audioBusy = true; $("audio-methods").setAttribute("aria-busy", "true"); $("audio-submit").disabled = true; $("audio-submit").textContent = "Comparando…";
  status("audio-status", "Extrayendo características y comparando el fragmento. El alineamiento temporal puede tardar más.", "busy");
  try {
    const result = await request("/api/audio/search", { method: "POST", body: form });
    const query = result.query || {}, url = localURL(query.audio_url, "/api/audio/");
    $("audio-query-summary").hidden = false;
    $("audio-query-description").textContent = `${number(query.duration || $("audio-duration").value, 1)} segundos${query.source_id !== undefined && query.source_id !== null ? " · Pista " + query.source_id : " · Archivo subido"}`;
    $("audio-query-player").hidden = !url; if (url) $("audio-query-player").src = url;
    $("audio-extraction").replaceChildren();
    Object.entries(result.extraction_ms || {}).forEach(([name, value]) => {
      const label = name.replace(/_/g, " ").replace(/ ms$/i, ""), item = element("span");
      item.append(document.createTextNode(label + ": "), element("strong", "", number(value, 2) + " ms")); $("audio-extraction").append(item);
    });
    $("audio-methods").replaceChildren(...(result.methods || []).map(renderAudioMethod));
    $("audio-corpus-count").textContent = `${number(result.corpus_count || 0)} pistas comparadas`;
    const scope = result.dtw_scope;
    $("audio-scope").hidden = !scope;
    $("audio-scope").textContent = typeof scope === "string" ? scope : scope ? "Alcance DTW: " + JSON.stringify(scope) : "";
    status("audio-status", "Comparación completada. Escucha los resultados y revisa sus métricas.");
  } catch (error) { status("audio-status", error.message, "error"); }
  finally { audioBusy = false; $("audio-methods").removeAttribute("aria-busy"); $("audio-submit").disabled = !audioReady; $("audio-submit").textContent = "Comparar fragmento"; }
});

async function loadExperiments() {
  const container = $("benchmark-images");
  container.replaceChildren(element("p", "form-hint busy", "Consultando resultados medidos…"));
  try {
    const payload = await request("/api/benchmarks"), items = Array.isArray(payload) ? payload : payload.items || [];
    container.replaceChildren();
    if (!items.length) { container.append(element("p", "form-hint", "Los gráficos aparecerán cuando finalicen los experimentos.")); return; }
    items.forEach((item) => {
      const url = localURL(item.url, "/benchmarks/"); if (!url) return;
      const figure = element("figure", "benchmark-figure"), link = element("a"), img = element("img");
      link.href = url; link.target = "_blank"; link.rel = "noopener";
      img.src = url; img.alt = item.name || "Gráfica experimental"; img.loading = "lazy";
      link.append(img); figure.append(link, element("figcaption", "", item.name || "Resultado experimental")); container.append(figure);
    });
  } catch (error) { container.replaceChildren(element("p", "status-message error", error.message)); }
}
