// Transport self-test, for the standalone /selftest page. Markup lives in
// flask_frontend/_selftest.html; every id is prefixed `st-`.
window.initSelftest = function (socket) {
  const $ = (i) => document.getElementById(i);
  let token = 0, n = 0, bytes = 0, t0 = null, arrivals = [], sendDone = null, nAtDone = 0;
  let running = false, rollHist = [], lastDemand = 0, pendingStart = null, lastCfg = null;
  // Once a run ends the gauges FREEZE at their last live reading. Frames still draining out of
  // the buffers are counted separately, so they cannot drag the measured rate toward zero.
  let frozen = false, finalW = null, drainFrames = 0, drainBytes = 0, drainLast = null;

  const showTransport = () => {
    try { $("st-t").textContent = socket.io.engine.transport.name; } catch (e) {}
  };
  socket.on("connect", () => {
    showTransport();
    if (pendingStart) { const p = pendingStart; pendingStart = null; setTimeout(() => doStart(p), 150); }
  });
  try { socket.io.engine.on("upgrade", showTransport); } catch (e) {}
  showTransport();

  socket.on("selftest_frame", (d) => {
    const buf = (d && d.buffer) ? d.buffer : d;
    const len = (d && (d.byteLength != null ? d.byteLength : (d.length || 0))) || 0;
    // Discard frames still draining from an abandoned run: they belong to the old settings.
    let tok = -1;
    try { tok = new DataView(buf, (d && d.byteOffset) || 0, 12).getUint32(4, true); } catch (e) {}
    if (tok !== token) return;
    if (frozen) { drainFrames++; drainBytes += len; drainLast = performance.now(); return; }
    if (t0 === null) t0 = performance.now();
    n++; bytes += len;
    arrivals.push({ t: performance.now(), b: len });
  });

  /** MB/s over a trailing window of `win` seconds (never longer than the run so far). */
  function windowRate(win) {
    if (t0 === null) return null;
    const now = performance.now();
    const span = Math.min(win * 1000, now - t0);
    if (span < 1500) return null;
    const cutoff = now - span;
    let b = 0, c = 0;
    for (let i = arrivals.length - 1; i >= 0; i--) {
      if (arrivals[i].t < cutoff) break;
      b += arrivals[i].b; c++;
    }
    return { mb: b / (span / 1000) / 1048576, fps: c / (span / 1000) };
  }

  setInterval(() => {
    if (frozen) { renderDrain(); return; }
    // On a timer, not per frame: while a run is live, delivery stopping dead must show as the
    // rate falling to zero. A per-frame update would sit at the last good value and hide it.
    const cut = performance.now() - 30000;
    while (arrivals.length && arrivals[0].t < cut) arrivals.shift();
    const w = windowRate(5);
    if (w) {
      $("st-roll").textContent = w.mb.toFixed(3);
      $("st-rollfps").textContent = w.fps.toFixed(1);
      rollHist.push(w.mb);
      if (rollHist.length > 6) rollHist.shift();
      if (rollHist.length >= 4) {
        const recent = rollHist.slice(-4), lo = Math.min(...recent), hi = Math.max(...recent);
        $("st-stable").innerHTML = (hi > 0 && (hi - lo) / hi < 0.20)
          ? '<span class="st-ok">stable</span>' : '<span class="st-warn">still moving</span>';
      }
    }
    if (t0 !== null) {
      const el = (performance.now() - t0) / 1000;
      $("st-n").textContent = n;
      $("st-f").textContent = (n / el).toFixed(1);
      $("st-mb").textContent = (bytes / el / 1048576).toFixed(3);
    }
  }, 500);

  function renderDrain() {
    if (sendDone === null) return;
    const since = (performance.now() - sendDone) / 1000;
    const quiet = drainLast === null ? since : (performance.now() - drainLast) / 1000;
    if (drainFrames === 0) {
      $("st-dr").textContent = quiet > 8 ? "nothing further arrived - buffers were empty" : "waiting…";
      return;
    }
    const mb = drainBytes / 1048576, secs = Math.max((drainLast - sendDone) / 1000, 0.001);
    $("st-dr").innerHTML = drainFrames + " frames (" + mb.toFixed(2) + " MB) over "
      + secs.toFixed(0) + " s = " + (mb / secs).toFixed(3) + " MB/s"
      + (quiet > 8 ? ' <span class="hint">(drain complete)</span>' : ' <span class="hint">…</span>');
  }

  function verdictFor(w) {
    if (!w) return { txt: "no data yet", cls: "hint" };
    const ratio = lastDemand > 0 ? (w.mb / lastDemand) : 0;
    if (ratio >= 0.9) return { txt: "KEEPS UP at " + w.mb.toFixed(3) + " MB/s", cls: "st-ok" };
    if (ratio >= 0.5) return { txt: "MARGINAL - " + (ratio * 100).toFixed(0) + "% of demand", cls: "st-warn" };
    const kb = (lastCfg && lastCfg.kb) || +$("st-kb").value;
    return { txt: "PIPE IS THE LIMIT - " + w.mb.toFixed(3) + " MB/s = "
      + (w.mb * 1048576 / 1024 / Math.max(kb, 1)).toFixed(1) + " fps at " + kb + " KB", cls: "st-bad" };
  }

  function record() {
    const w = finalW || windowRate(5), v = verdictFor(w);
    $("st-r").innerHTML = '<span class="' + v.cls + '">' + v.txt + "</span>";
    if (!w || !lastCfg) return;      // stopped before any data: no row worth comparing
    const row = $("st-hist").insertRow(-1);
    row.innerHTML = "<td>" + lastCfg.kb + "</td><td>" + lastCfg.fps + "</td><td>"
      + lastDemand.toFixed(2) + "</td><td><b>" + w.mb.toFixed(3) + "</b></td><td>"
      + w.fps.toFixed(1) + '</td><td class="' + v.cls + '">' + v.txt + "</td>";
  }

  /** End the run: keep the last live reading, and stop the gauges moving. */
  function freeze(why) {
    if (frozen) return;
    finalW = windowRate(5) || finalW;
    frozen = true; running = false;
    sendDone = performance.now(); nAtDone = n;
    drainFrames = 0; drainBytes = 0; drainLast = null;
    $("st-go").disabled = false; $("st-stop").disabled = true;
    $("st-stable").innerHTML = '<span class="hint">final (' + why + ")</span>";
    if (finalW) {
      $("st-roll").textContent = finalW.mb.toFixed(3);
      $("st-rollfps").textContent = finalW.fps.toFixed(1);
    }
    record();
  }

  socket.on("selftest_done", (d) => {
    if (d.token !== token) return;      // an abandoned run finishing; not ours
    $("st-nnote").textContent = "(sender emitted " + d.emitted + ")";
    // On a slow link this arrives only once the backlog has drained, so the run is long over.
    freeze(d.stopped ? "stopped" : "finished");
  });

  function doStart(cfg) {
    token = (token + 1) & 0xffffffff;
    n = 0; bytes = 0; t0 = null; arrivals = []; rollHist = []; sendDone = null; nAtDone = 0;
    running = true; frozen = false; finalW = null;
    drainFrames = 0; drainBytes = 0; drainLast = null;
    lastCfg = cfg; lastDemand = cfg.fps * cfg.kb * 1024 / 1048576;
    $("st-want").textContent = cfg.fps + " fps x " + cfg.kb + " KB = " + lastDemand.toFixed(2) + " MB/s";
    $("st-r").textContent = "running..."; $("st-dr").textContent = "-";
    $("st-nnote").textContent = ""; $("st-roll").textContent = "-";
    $("st-rollfps").textContent = "-"; $("st-stable").textContent = "-";
    $("st-go").disabled = true; $("st-stop").disabled = false;
    socket.emit("selftest_start", { fps: cfg.fps, seconds: cfg.seconds, kb: cfg.kb, token: token });
  }

  function clearAll() {
    token = (token + 1) & 0xffffffff;      // orphan any frames still in flight
    frozen = false; running = false; finalW = null; lastCfg = null;
    n = 0; bytes = 0; t0 = null; arrivals = []; rollHist = [];
    sendDone = null; drainFrames = 0; drainBytes = 0; drainLast = null;
    $("st-go").disabled = false; $("st-stop").disabled = true;
    ["st-roll", "st-rollfps", "st-n", "st-f", "st-mb"].forEach((i) => { $(i).textContent = "-"; });
    $("st-stable").textContent = "-"; $("st-r").textContent = "-"; $("st-nnote").textContent = "";
  }

  $("st-go").onclick = () => {
    const cfg = { fps: +$("st-fps").value, kb: +$("st-kb").value, seconds: +$("st-secs").value };
    // A previous run still draining has its frames queued ahead of the new run's, so the new
    // measurement would read the old backlog. Only a fresh connection discards it.
    const draining = sendDone !== null && (performance.now() - sendDone) < 20000;
    if (running || draining) {
      pendingStart = cfg; socket.disconnect(); socket.connect();
    } else {
      doStart(cfg);
    }
  };
  $("st-stop").onclick = () => {
    // Freeze first, then tell the server: the reading must be the one from while frames were
    // still flowing, not whatever the tail of the drain looks like.
    freeze("stopped");
    socket.emit("selftest_stop");
  };
  $("st-reset").onclick = () => {
    socket.emit("selftest_stop");
    clearAll();
    socket.disconnect(); socket.connect();
    $("st-dr").textContent = "link reset - buffers discarded, history kept";
  };
  document.querySelectorAll("button.st-mini").forEach((b) => {
    b.onclick = () => {
      const p = b.dataset.p.split(",");
      $("st-kb").value = p[0]; $("st-fps").value = p[1];
    };
  });
};
