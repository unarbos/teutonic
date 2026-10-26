(function (root) {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";
  function svgElement(tag, attributes, text) {
    var node = document.createElementNS(NS, tag);
    Object.keys(attributes || {}).forEach(function (key) { node.setAttribute(key, attributes[key]); });
    if (text != null) node.textContent = text;
    return node;
  }
  function element(tag, text) {
    var node = document.createElement(tag);
    if (text != null) node.textContent = text;
    return node;
  }
  function percent(value) { return (value * 100).toFixed(4).replace(/\.?0+$/, "") + "%"; }
  function overlap(a, b) {
    var weight = 0;
    a.weights.forEach(function (value, key) { weight += Math.min(value, b.weights.get(key) || 0); });
    return weight;
  }
  function compareMixes(a, b) {
    var shared = [], onlyFrom = [], onlyTo = [];
    a.categories.forEach(function (_, key) { (b.categories.has(key) ? shared : onlyFrom).push(key); });
    b.categories.forEach(function (_, key) { if (!a.categories.has(key)) onlyTo.push(key); });
    return { from: a.key, to: b.key, shared: shared, onlyFrom: onlyFrom, onlyTo: onlyTo, weight: overlap(a, b) };
  }
  function buildGraph(nodes) {
    var links = [];
    nodes.forEach(function (a, index) {
      nodes.slice(index + 1).forEach(function (b) {
        var link = compareMixes(a, b);
        if (link.shared.length) links.push(link);
      });
    });
    return links;
  }
  var api = {
    buildGraph: buildGraph,
    compareMixes: compareMixes,
    create: function () {
      var container = document.getElementById("competition-shared-tree");
      var states = new Map(), lastSignature = "";
      var details = document.getElementById("competition-tree-details");
      var title = document.getElementById("tree-details-title");
      var summary = document.getElementById("tree-details-summary");
      var filters = document.getElementById("tree-details-filters");
      var columns = document.getElementById("tree-details-columns");
      var rows = document.getElementById("tree-details-rows");
      var currentNodes = new Map(), selection = null;
      var requestedRoot = new URL(window.location.href).searchParams.get("competition");
      var rootKey = /^(main|mix[0-9]{2})$/.test(requestedRoot || "") ? requestedRoot : "mix06";
      var categoryButton = document.getElementById("tree-root-categories");
      var comparePicker = document.getElementById("tree-compare-picker");
      var svg = null, edgeLayer = null, nodeLayer = null, animationFrame = null, layoutFrame = null;
      var positions = new Map(), nodeViews = new Map(), edgeViews = new Map(), highlightKey = null;
      var animationDeadline = 0, graphRevision = 0, lastLayout = "", lastPicker = "";
      var reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
      var compactLayout = window.matchMedia("(max-width: 640px)");
      var observedWidth = container.clientWidth;
      var pendingWidth = null;
      var tooltip = element("div"), tooltipBounds = null, tooltipSize = null;
      tooltip.className = "competition-edge-tooltip"; tooltip.hidden = true; tooltip.setAttribute("role", "tooltip");
      document.body.appendChild(tooltip);
      function hideTooltip() { tooltip.hidden = true; }
      function positionTooltip(x, y) {
        if (tooltip.hidden || !tooltipBounds) return;
        var left = Math.max(tooltipBounds.left, Math.min(x + 12, tooltipBounds.right - tooltipSize.width));
        var preferredTop = y - tooltipSize.height - 10;
        if (preferredTop < tooltipBounds.top) preferredTop = y + 12;
        var top = Math.max(tooltipBounds.top, Math.min(preferredTop, tooltipBounds.bottom - tooltipSize.height));
        tooltip.style.transform = "translate3d(" + left + "px," + top + "px,0)";
      }
      function showTooltip(view, event) {
        if (svg.classList.contains("is-moving")) return;
        tooltip.textContent = view.label.textContent;
        tooltip.hidden = false;
        tooltipSize = tooltip.getBoundingClientRect();
        var bounds = container.getBoundingClientRect();
        tooltipBounds = { left: Math.max(8, bounds.left + 4), right: Math.min(window.innerWidth - 8, bounds.right - 4), top: Math.max(8, bounds.top + 4), bottom: Math.min(window.innerHeight - 8, bounds.bottom - 4) };
        var lineBounds = view.line.getBoundingClientRect();
        positionTooltip(event && event.clientX != null ? event.clientX : lineBounds.left + lineBounds.width / 2, event && event.clientY != null ? event.clientY : lineBounds.top + lineBounds.height / 2);
      }
      function refreshFocusedTooltip() {
        var focused = document.activeElement;
        var view = focused && edgeViews.get(focused.dataset.from + ":" + focused.dataset.to);
        if (view && focused === view.group) showTooltip(view); else hideTooltip();
      }
      window.addEventListener("scroll", refreshFocusedTooltip, true);
      window.addEventListener("resize", hideTooltip);
      function scheduleLayout(width) {
        if (!currentNodes.size) return;
        pendingWidth = typeof width === "number" ? width : null;
        if (layoutFrame != null) return;
        layoutFrame = requestAnimationFrame(function () { layoutFrame = null; var width = pendingWidth; pendingWidth = null; drawTree(false, width); });
      }
      if (typeof ResizeObserver !== "undefined") {
        new ResizeObserver(function (entries) {
          var width = Math.round(entries[0].contentRect.width);
          if (width && width !== observedWidth) scheduleLayout(width);
        }).observe(container);
      } else window.addEventListener("resize", scheduleLayout);

      function matchesSelection(node) {
        if (!selection) return false;
        if (!selection.other) return node.dataset.competition === selection.key;
        return (node.dataset.from === selection.key && node.dataset.to === selection.other) || (node.dataset.to === selection.key && node.dataset.from === selection.other);
      }
      function markSelection() {
        container.querySelectorAll("[data-selection]").forEach(function (node) {
          var isCompetition = node.classList.contains("competition-tree-node");
          node.setAttribute("aria-pressed", String(isCompetition ? node.dataset.competition === rootKey : matchesSelection(node)));
        });
      }
      function highlightConnections(key) {
        highlightKey = key;
        if (!svg) return;
        var active = key || rootKey;
        svg.dataset.highlight = active;
        edgeLayer.querySelectorAll(".competition-tree-edge").forEach(function (edge) {
          edge.classList.toggle("is-highlighted", edge.dataset.from === active || edge.dataset.to === active);
        });
      }
      function closeDetails() {
        var trigger = Array.from(container.querySelectorAll("[data-selection]")).find(matchesSelection);
        details.hidden = true; selection = null; comparePicker.value = ""; markSelection();
        if (trigger) trigger.focus(); else comparePicker.focus();
      }
      document.getElementById("tree-details-close").addEventListener("click", closeDetails);
      function escape(event) { if (event.key === "Escape" && selection) { event.preventDefault(); closeDetails(); } }
      details.addEventListener("keydown", escape); container.addEventListener("keydown", escape);
      filters.addEventListener("click", function (event) {
        var button = event.target.closest("button[data-view]");
        if (!button || !selection || !selection.other) return;
        selection.view = button.dataset.view;
        renderDetails(false);
        details.querySelector(".tree-details-scroll").scrollTop = 0;
      });
      function renderDetails(focus) {
        if (!selection) return;
        var a = currentNodes.get(selection.key), b = currentNodes.get(selection.other);
        if (!a || (selection.other && !b)) { closeDetails(); return; }
        comparePicker.value = b && a.key === rootKey ? b.key : b && b.key === rootKey ? a.key : "";
        var all = new Map(a.categories);
        if (b) b.categories.forEach(function (category, key) { if (!all.has(key)) all.set(key, category); });
        var comparison = b ? compareMixes(a, b) : null;
        var sharedCount = comparison ? comparison.shared.length : 0;
        var view = selection.view || "shared";
        var visible = Array.from(all).filter(function (entry) {
          if (!b || view === "all") return true;
          if (view === "only-from") return a.categories.has(entry[0]) && !b.categories.has(entry[0]);
          if (view === "only-to") return b.categories.has(entry[0]) && !a.categories.has(entry[0]);
          return a.categories.has(entry[0]) && b.categories.has(entry[0]);
        });
        visible.sort(function (left, right) {
          function weight(entry) {
            var first = a.weights.get(entry[0]) || 0, second = b ? b.weights.get(entry[0]) || 0 : 0;
            return b && view === "shared" ? Math.min(first, second) : Math.max(first, second);
          }
          return weight(right) - weight(left) || left[1].name.localeCompare(right[1].name);
        });
        title.textContent = a.key.toUpperCase() + (b ? " ↔ " + b.key.toUpperCase() : " · ALL CATEGORIES");
        summary.textContent = b ? sharedCount + " shared categories · " + comparison.onlyFrom.length + " only in " + a.key.toUpperCase() + " · " + comparison.onlyTo.length + " only in " + b.key.toUpperCase() + " · " + (comparison.weight * 100).toFixed(1) + "% shared sampling weight" : all.size + " categories · full competition mix";
        filters.hidden = !b;
        filters.querySelectorAll("button").forEach(function (button) {
          var labels = { shared: "SHARED · " + sharedCount, all: "ALL CATEGORIES · " + all.size };
          if (b) {
            labels["only-from"] = "ONLY " + a.key.toUpperCase() + " · " + comparison.onlyFrom.length;
            labels["only-to"] = "ONLY " + b.key.toUpperCase() + " · " + comparison.onlyTo.length;
          }
          button.textContent = labels[button.dataset.view] || "";
          button.setAttribute("aria-pressed", String(button.dataset.view === view));
        });
        columns.replaceChildren();
        var heading = element("tr");
        ["CATEGORY", a.key.toUpperCase()].concat(b ? [b.key.toUpperCase()] : []).forEach(function (label) {
          var th = element("th", label); th.scope = "col"; heading.appendChild(th);
        });
        columns.appendChild(heading); rows.replaceChildren();
        visible.forEach(function (entry) {
          var row = element("tr"), category = entry[1], name = element("td");
          row.dataset.category = category.name; row.dataset.dataset = category.dataset;
          name.appendChild(element("span", category.name)); name.appendChild(element("small", category.dataset)); row.appendChild(name);
          [a].concat(b ? [b] : []).forEach(function (competition) {
            row.appendChild(element("td", competition.weights.has(entry[0]) ? percent(competition.weights.get(entry[0])) : "—"));
          });
          rows.appendChild(row);
        });
        if (!visible.length) {
          var empty = element("tr"), cell = element("td", view === "shared" ? "No shared categories" : "No categories in this selection"); cell.colSpan = b ? 3 : 2; empty.appendChild(cell); rows.appendChild(empty);
        }
        details.hidden = false; markSelection();
        if (focus) title.focus({ preventScroll: window.matchMedia("(min-width: 1200px)").matches });
      }
      function showCategories() {
        if (!currentNodes.has(rootKey)) return;
        selection = { key: rootKey }; renderDetails(true);
        details.querySelector(".tree-details-scroll").scrollTop = 0;
      }
      categoryButton.addEventListener("click", showCategories);
      comparePicker.addEventListener("change", function () {
        if (!currentNodes.has(comparePicker.value) || comparePicker.value === rootKey) return;
        selection = { key: rootKey, other: comparePicker.value, view: "shared" };
        renderDetails(true); details.querySelector(".tree-details-scroll").scrollTop = 0;
      });
      function pickRoot(key) {
        if (!currentNodes.has(key)) return;
        var changed = key !== rootKey;
        rootKey = key;
        var url = new URL(window.location.href);
        url.searchParams.set("competition", rootKey);
        window.history.replaceState(null, "", url);
        selection = { key: rootKey }; renderDetails(!changed);
        // Measure once after the details pane has taken its final width.
        if (changed) drawTree(true);
        details.querySelector(".tree-details-scroll").scrollTop = 0;
        if (changed) container.scrollTo({ left: 0, behavior: reducedMotion.matches ? "auto" : "smooth" });
      }
      function activate(group, value, label, action) {
        group.setAttribute("role", "button"); group.setAttribute("tabindex", "0");
        group.setAttribute("aria-label", label); group.setAttribute("aria-controls", details.id);
        group.setAttribute("aria-pressed", "false");
        group.dataset.selection = value.key + (value.other ? ":" + value.other : "");
        function open() {
          if (action) { action(); return; }
          selection = Object.assign({}, value); renderDetails(true);
          details.querySelector(".tree-details-scroll").scrollTop = 0;
        }
        group.addEventListener("click", open);
        group.addEventListener("keydown", function (event) {
          if (event.key === "Enter" || event.key === " ") { event.preventDefault(); open(); }
        });
      }

      function drawTree(animate, measuredWidth) {
        if (layoutFrame != null) { cancelAnimationFrame(layoutFrame); layoutFrame = null; pendingWidth = null; }
        if (measuredWidth == null) measuredWidth = container.clientWidth;
        observedWidth = measuredWidth;
        var compact = compactLayout.matches;
        var layout = [rootKey, measuredWidth, compact, graphRevision].join(":");
        if (layout === lastLayout) return;
        lastLayout = layout;
        hideTooltip();
        var now = performance.now();
        var duration = animate ? 420 : animationFrame != null ? Math.max(0, animationDeadline - now) : 0;
        if (animationFrame != null) cancelAnimationFrame(animationFrame);
        animationFrame = null;
        var root = currentNodes.get(rootKey), nodes = Array.from(currentNodes.values());
        var links = buildGraph(nodes);
        var width = compact ? 420 : Math.max(740, measuredWidth);
        var height = compact ? 440 : 340;
        var center = { x: width / 2, y: height / 2 }, targets = new Map();
        targets.set(rootKey, { x: center.x, y: center.y, emphasis: 1 });
        var peers = nodes.filter(function (node) { return node.key !== rootKey; });
        peers.forEach(function (node, index) {
          var angle = -Math.PI / 2 + index * Math.PI * 2 / peers.length;
          targets.set(node.key, { x: center.x + Math.cos(angle) * (compact ? 126 : (width - 180) / 2), y: center.y + Math.sin(angle) * (compact ? 160 : 120), emphasis: 0 });
        });
        var initial = !svg;
        if (!svg) {
          svg = svgElement("svg", { class: "competition-graph", role: "group", "aria-labelledby": "competition-tree-title competition-tree-description" });
          svg.appendChild(svgElement("title", { id: "competition-tree-title" }, "Competition shared-category graph"));
          svg.appendChild(svgElement("desc", { id: "competition-tree-description" }));
          edgeLayer = svgElement("g", { class: "competition-tree-edges" });
          nodeLayer = svgElement("g", { class: "competition-tree-nodes" });
          svg.appendChild(edgeLayer); svg.appendChild(nodeLayer); container.replaceChildren(svg);
        }
        var previousSize = initial ? { width: width, height: height } : { width: svg.viewBox.baseVal.width, height: svg.viewBox.baseVal.height };
        if (previousSize.width !== width || previousSize.height !== height) positions.forEach(function (point) {
          point.x *= width / previousSize.width; point.y *= height / previousSize.height;
        });
        svg.setAttribute("viewBox", "0 0 " + width + " " + height);
        svg.dataset.focus = rootKey;
        svg.querySelector("desc").textContent = rootKey.toUpperCase() + " is focused. Every pair sharing at least one category has a connection. Hover or focus a competition to highlight its connections, click it to move it to the center, or select any line to compare shared and unique categories. Percentages measure shared sampling weight.";
        categoryButton.disabled = false;
        categoryButton.textContent = "VIEW " + rootKey.toUpperCase() + " CATEGORIES · " + root.categories.size;
        var pickerSignature = rootKey + ":" + nodes.map(function (node) { return node.key; }).join(":");
        if (pickerSignature !== lastPicker) {
          lastPicker = pickerSignature;
          comparePicker.replaceChildren(element("option", "CHOOSE COMPETITION"));
          comparePicker.firstChild.value = "";
          nodes.filter(function (node) { return node.key !== rootKey; }).sort(function (a, b) { return a.key.localeCompare(b.key); }).forEach(function (node) {
            var option = element("option", node.key.toUpperCase()); option.value = node.key; comparePicker.appendChild(option);
          });
        }
        comparePicker.disabled = false;
        comparePicker.value = selection && selection.other ? selection.key === rootKey ? selection.other : selection.other === rootKey ? selection.key : "" : "";
        currentNodes.forEach(function (node) {
          var view = nodeViews.get(node.key);
          if (!view) {
            var group = svgElement("g", { class: "competition-tree-node", "data-competition": node.key });
            var box = svgElement("rect", {});
            var caption = svgElement("text", { class: "competition-root-caption", "text-anchor": "middle" }, "FOCUS");
            var name = svgElement("text", { class: "competition-node-name", "text-anchor": "middle" }, node.key.toUpperCase());
            var count = svgElement("text", { class: "competition-node-count", "text-anchor": "middle" });
            group.appendChild(box); group.appendChild(caption); group.appendChild(name); group.appendChild(count);
            activate(group, { key: node.key }, "Focus " + node.key.toUpperCase() + " and show its categories", function () { pickRoot(node.key); });
            group.addEventListener("pointerenter", function () { highlightConnections(node.key); });
            group.addEventListener("pointerleave", function () { highlightConnections(null); });
            group.addEventListener("focus", function () { highlightConnections(node.key); });
            group.addEventListener("blur", function () { highlightConnections(null); });
            nodeLayer.appendChild(group);
            view = { group: group, box: box, caption: caption, name: name, count: count };
            nodeViews.set(node.key, view);
          }
          view.group.classList.toggle("is-root", node.key === rootKey);
          view.group.setAttribute("aria-current", node.key === rootKey ? "true" : "false");
          var emphasis = node.key === rootKey ? 1 : 0, boxWidth = 110 + 18 * emphasis, boxHeight = 44 + 20 * emphasis;
          view.box.setAttribute("x", -boxWidth / 2); view.box.setAttribute("y", -boxHeight / 2);
          view.box.setAttribute("width", boxWidth); view.box.setAttribute("height", boxHeight);
          view.caption.setAttribute("y", -19); view.caption.style.opacity = emphasis;
          view.name.setAttribute("y", -2 + emphasis * 7); view.count.setAttribute("y", 14 + emphasis * 8);
          if (view.count.textContent !== node.categories.size + " categories") view.count.textContent = node.categories.size + " categories";
          view.group.setAttribute("aria-label", "Focus " + node.key.toUpperCase() + ": " + node.categories.size + " categories. Show all categories.");
          if (!positions.has(node.key)) positions.set(node.key, Object.assign({}, targets.get(node.key)));
        });
        nodeViews.forEach(function (view, key) {
          if (!currentNodes.has(key)) { view.group.remove(); nodeViews.delete(key); positions.delete(key); }
        });
        var activeEdges = new Set();
        var edges = links.map(function (edge) {
          var key = edge.from + ":" + edge.to;
          activeEdges.add(key);
          var view = edgeViews.get(key);
          var weight = edge.weight, shared = edge.shared.length;
          if (!view) {
            var group = svgElement("g", { class: "competition-tree-edge", "data-from": edge.from, "data-to": edge.to });
            activate(group, { key: edge.from, other: edge.to, view: "shared" }, "Show comparison");
            var hit = svgElement("path", { class: "competition-tree-hit" }), line = svgElement("path", { class: "competition-tree-line" });
            var labels = svgElement("g", { class: "competition-edge-labels" });
            var overlapLabel = svgElement("text", { "text-anchor": "middle", y: -4, class: "competition-graph-overlap" });
            labels.appendChild(overlapLabel);
            group.appendChild(hit); group.appendChild(line); group.appendChild(labels);
            edgeLayer.appendChild(group);
            view = { from: edge.from, to: edge.to, group: group, hit: hit, line: line, labels: labels, label: overlapLabel };
            edgeViews.set(key, view);
            group.addEventListener("pointerenter", function (event) { showTooltip(view, event); });
            group.addEventListener("pointermove", function (event) { if (tooltip.hidden) showTooltip(view, event); else positionTooltip(event.clientX, event.clientY); });
            group.addEventListener("pointerleave", hideTooltip);
            group.addEventListener("focus", function () { showTooltip(view); });
            group.addEventListener("blur", hideTooltip);
          }
          var signature = [shared, edge.onlyFrom.length, edge.onlyTo.length, weight].join(":");
          if (view.signature !== signature) {
            view.signature = signature;
            view.group.dataset.shared = shared; view.group.dataset.onlyFrom = edge.onlyFrom.length;
            view.group.dataset.onlyTo = edge.onlyTo.length; view.group.dataset.overlap = weight;
            var relation = edge.from.toUpperCase() + " and " + edge.to.toUpperCase() + " share " + shared + " categories";
            view.group.setAttribute("aria-label", relation + ". Show comparison.");
            view.label.textContent = (weight * 100).toFixed(1) + "% overlap";
          }
          return view;
        });
        edgeViews.forEach(function (view, key) { if (!activeEdges.has(key)) { view.group.remove(); edgeViews.delete(key); } });
        highlightConnections(highlightKey && currentNodes.has(highlightKey) ? highlightKey : null);
        function boundary(point, toward) {
          var dx = toward.x - point.x, dy = toward.y - point.y;
          var scale = 1 / Math.max(Math.abs(dx) / (58 + 9 * point.emphasis), Math.abs(dy) / (25 + 10 * point.emphasis), 1);
          return { x: point.x + dx * scale, y: point.y + dy * scale };
        }
        function paint(updateLabels) {
          nodeViews.forEach(function (view, key) {
            var point = positions.get(key);
            view.group.setAttribute("transform", "translate(" + point.x + " " + point.y + ")");
          });
          edges.forEach(function (edge) {
            var origin = positions.get(edge.from), target = positions.get(edge.to);
            var control = { x: (origin.x + target.x) / 2, y: (origin.y + target.y) / 2 };
            if (edge.from !== rootKey && edge.to !== rootKey) {
              var dx = control.x - center.x, dy = control.y - center.y, distance = Math.hypot(dx, dy);
              var bulge = distance < 70 ? 180 : 45;
              if (distance < 10) { dx = -(target.y - origin.y); dy = target.x - origin.x; distance = Math.hypot(dx, dy); }
              if (distance) { control.x += dx / distance * bulge; control.y += dy / distance * bulge; }
            }
            var start = boundary(origin, control), end = boundary(target, control);
            var path = "M" + start.x + "," + start.y + " Q" + control.x + "," + control.y + " " + end.x + "," + end.y;
            var labelX = Math.max(60, Math.min(width - 60, (start.x + 2 * control.x + end.x) / 4));
            var labelY = Math.max(22, Math.min(height - 24, (start.y + 2 * control.y + end.y) / 4));
            edge.hit.setAttribute("d", path); edge.line.setAttribute("d", path);
            if (updateLabels) edge.labels.setAttribute("transform", "translate(" + labelX + " " + labelY + ")");
          });
        }
        markSelection();
        if (!duration || reducedMotion.matches || initial) {
          animationDeadline = 0; svg.classList.remove("is-moving");
          positions = targets; paint(true); refreshFocusedTooltip();
          if (initial && !reducedMotion.matches && svg.animate) svg.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 250 });
          return;
        }
        var starts = new Map();
        positions.forEach(function (point, key) { starts.set(key, Object.assign({}, point)); });
        var startedAt = performance.now();
        animationDeadline = startedAt + duration;
        svg.classList.add("is-moving");
        paint();
        function frame(now) {
          var progress = Math.min(1, (now - startedAt) / duration), eased = 1 - Math.pow(1 - progress, 3);
          targets.forEach(function (target, key) {
            var start = starts.get(key);
            positions.set(key, {
              x: start.x + (target.x - start.x) * eased,
              y: start.y + (target.y - start.y) * eased,
              emphasis: start.emphasis + (target.emphasis - start.emphasis) * eased,
            });
          });
          if (progress === 1) positions = targets;
          paint(progress === 1);
          if (progress < 1) animationFrame = requestAnimationFrame(frame);
          else { animationFrame = null; animationDeadline = 0; svg.classList.remove("is-moving"); refreshFocusedTooltip(); }
        }
        animationFrame = requestAnimationFrame(frame);
      }

      function render() {
        var active = Array.from(states.values()).filter(function (state) { return state.active !== false; });
        active.sort(function (a, b) { return a.key === b.key ? 0 : a.key === "main" ? -1 : b.key === "main" ? 1 : a.key.localeCompare(b.key); });
        var signature = JSON.stringify(active.map(function (state) { return [state.key, state.dataset && state.dataset.sources, state.datasetFailed]; }));
        if (signature === lastSignature) return;
        lastSignature = signature;
        if (!active.length || active.some(function (state) { return !state.dataset; })) {
          if (!svg) container.textContent = "Loading public competition data…";
          return;
        }
        var nodes = active.map(function (state) {
          var sources = state.dataset.sources.filter(function (source) { return source && source.enabled !== false; });
          var weights = new Map(), categories = new Map();
          TeutonicDashboardV1.datasetPresentation(state.dataset).rows.forEach(function (row, index) {
            var identity = (sources[index].dataset || sources[index].source || "") + "\0" + row.name;
            weights.set(identity, row.normalizedWeight);
            categories.set(identity, { name: row.name, dataset: sources[index].dataset || sources[index].source || "Unspecified dataset" });
          });
          return { key: state.key, weights: weights, categories: categories };
        });
        currentNodes = new Map(nodes.map(function (node) { return [node.key, node]; }));
        graphRevision++;
        if (!currentNodes.has(rootKey)) rootKey = currentNodes.has("main") ? "main" : nodes[0].key;
        drawTree(false);
        renderDetails(false);
        if (active.some(function (state) { return state.datasetFailed; })) {
          var status = document.createElement("p"); status.textContent = "Refresh unavailable · showing last published data"; container.appendChild(status);
        }
      }
      return {
        update: function (state) { states.set(state.key, state); render(); },
        unavailable: function () {
          if (!container.querySelector("svg")) container.textContent = "Competition data unavailable. Retrying on the next refresh.";
        }
      };
    }
  };
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.TeutonicCompetitionTree = api;
})(typeof window !== "undefined" ? window : globalThis);
