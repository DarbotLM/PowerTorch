const byId = (id) => document.getElementById(id);

async function loadStatus() {
  try {
    const response = await fetch("./status.json", { cache: "no-store" });
    if (!response.ok) {
      return;
    }
    const status = await response.json();
    byId("test-total").textContent = String(status.xunitTests + status.pesterTests);
    byId("build-errors").textContent = `${status.buildErrors} / ${status.buildWarnings}`;
    byId("tensor-count").textContent = status.safetensorsTensors.toLocaleString();
    byId("commit-hash").textContent = `${status.commitShort} · ${status.generatedUtc.slice(0, 10)}`;
  } catch {
    // Static fallback values remain visible when the site is opened from file://.
  }
}

function statusBadge(value) {
  const normalized = String(value || "planned").toLowerCase();
  const label = normalized.charAt(0).toUpperCase() + normalized.slice(1);
  return `<span class="status ${normalized}">${label}</span>`;
}

async function loadCompatibility() {
  const target = byId("compatibility-table");
  try {
    const response = await fetch("./compatibility.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const document = await response.json();
    const rows = document.families.flatMap((family) =>
      family.capabilities.map((capability) => `
        <tr>
          <td>${escapeHtml(family.family)}</td>
          <td>${escapeHtml(capability.name)}</td>
          <td>${statusBadge(capability.status)}</td>
          <td>${capability.backends.map(escapeHtml).join(", ")}</td>
        </tr>
      `)
    );
    target.innerHTML = `
      <table>
        <thead>
          <tr><th>Family</th><th>Capability</th><th>Status</th><th>Backends</th></tr>
        </thead>
        <tbody>${rows.join("")}</tbody>
      </table>
    `;
  } catch (error) {
    target.innerHTML = `
      <p style="padding:18px;color:var(--muted)">
        Compatibility data could not be loaded (${escapeHtml(error.message)}).
        The source matrix is available in the repository.
      </p>
    `;
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

loadStatus();
loadCompatibility();
