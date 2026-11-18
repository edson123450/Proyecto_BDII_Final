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
    /('(?:''|[^'])*'|\b(?:SELECT|FROM|WHERE|AND|OR|INSERT|INTO|VALUES|CREATE|TABLE|INDEX|ON|USING|HEAP|SEQUENTIAL|BTREE|HASH|DELETE|PRIMARY|KEY|INT|CHAR|FLOAT|LIMIT|ORDER|BY|ASC|DESC)\b|\b\d+(?:\.\d+)?\b)/gi,
  );
  $("highlight").innerHTML =
    tokens
      .map((t) =>
        /^'/.test(t)
          ? `<span class="string">${escapeHTML(t)}</span>`
          : /^\d/.test(t)
            ? `<span class="number">${escapeHTML(t)}</span>`
            : /^[A-Z]+$/i.test(t) &&
                /^(SELECT|FROM|WHERE|AND|OR|INSERT|INTO|VALUES|CREATE|TABLE|INDEX|ON|USING|HEAP|SEQUENTIAL|BTREE|HASH|DELETE|PRIMARY|KEY|INT|CHAR|FLOAT|LIMIT|ORDER|BY|ASC|DESC)$/i.test(
                  t,
                )
              ? `<span class="kw">${escapeHTML(t)}</span>`
              : escapeHTML(t),
      )
      .join("") + "\n";
}
async function request(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail),
    );
  return data;
}
async function loadTables() {
  try {
    const tables = await request("/api/tables");
    $("tables").innerHTML = tables.length
      ? tables
          .map(
            (t) =>
              `<div class="table-card"><button class="table-name" data-table="${escapeHTML(t.name)}">▤ ${escapeHTML(t.name)}</button><div class="table-sub"><span class="chip">${escapeHTML(t.organization)}</span><span>${Number(t.record_count).toLocaleString("es-PE")} filas</span></div>${t.columns.map((c) => `<div class="column"><span>${c.primary ? "⌑ " : ""}${escapeHTML(c.name)}</span><code>${escapeHTML(c.type)}${c.size ? "(" + c.size + ")" : ""}</code></div>`).join("")}${t.indexes.map((i) => `<div class="index-item">↳ ${escapeHTML(i.name)} · ${escapeHTML(i.kind)}</div>`).join("")}${t.organization === "SEQUENTIAL" ? `<button class="secondary reorganize" data-reorganize="${escapeHTML(t.name)}">Reorganizar</button>` : ""}</div>`,
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
  $("plan-name").textContent = p.operation;
  $("plan-reason").textContent =
    p.reason || "Operación completada sobre los archivos del motor.";
  $("plan-index").textContent = p.index || "";
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
    total = result.total;
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
$("show-benchmarks").onclick = async () => {
  const items = await request("/api/benchmarks");
  $("benchmark-panel").hidden = false;
  $("benchmark-images").innerHTML = items.length
    ? items
        .map(
          (i) =>
            `<a href="${escapeHTML(i.url)}" target="_blank" rel="noopener"><img src="${escapeHTML(i.url)}" alt="${escapeHTML(i.name)}"></a>`,
        )
        .join("")
    : "<p>Los gráficos aparecerán después de ejecutar los experimentos.</p>";
  $("benchmark-panel").scrollIntoView({ behavior: "smooth" });
};
$("close-benchmarks").onclick = () => {
  $("benchmark-panel").hidden = true;
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
