// planner-charts.js — Plotly chart rendering for DBM-3 Dive Planner

var PlannerCharts = (function () {
  'use strict';

  var DARK_LAYOUT = {
    paper_bgcolor: '#1a1a2e',
    plot_bgcolor: '#0f0f1a',
    font: { color: '#e0e0e0', family: '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif' },
    margin: { t: 40, r: 20, b: 50, l: 60 },
    xaxis: { gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
    yaxis: { gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
  };

  var PLOTLY_CONFIG = { responsive: true, displayModeBar: false };

  function _isMobile() {
    return window.innerWidth < 600;
  }

  function _mobileMargin(defaults) {
    if (!_isMobile()) return defaults;
    return { t: Math.min(defaults.t, 25), r: 10, b: 40, l: 40 };
  }

  function _mobileLegend(defaults) {
    if (!_isMobile()) return defaults;
    return {
      orientation: 'h',
      x: 0.5,
      xanchor: 'center',
      y: -0.25,
      bgcolor: 'rgba(26, 26, 46, 0.8)',
      font: { size: 9 },
    };
  }

  function renderProfileChart(profileData, buhlmannCeilings, slabCeilings) {
    var elementId = 'profile-chart';
    Plotly.purge(elementId);

    var times = profileData.times;
    var depths = profileData.depths;

    // Depth profile trace (y-inverted: deeper = lower)
    var depthTrace = {
      x: times,
      y: depths.map(function (d) { return -d; }),
      name: 'Depth',
      type: 'scatter',
      mode: 'lines',
      fill: 'tozeroy',
      fillcolor: 'rgba(79, 195, 247, 0.1)',
      line: { color: '#4fc3f7', width: 2 },
      hovertemplate: 'Time: %{x:.1f} min<br>Depth: %{customdata:.1f} m<extra></extra>',
      customdata: depths,
    };

    var traces = [depthTrace];

    // Buhlmann ceiling line (if any ceiling > 0)
    if (buhlmannCeilings && buhlmannCeilings.some(function (c) { return c > 0; })) {
      var ceilTimes = [];
      var step = times.length / buhlmannCeilings.length;
      for (var i = 0; i < buhlmannCeilings.length; i++) {
        ceilTimes.push(times[Math.min(Math.floor(i * step), times.length - 1)]);
      }
      traces.push({
        x: ceilTimes,
        y: buhlmannCeilings.map(function (c) { return -c; }),
        name: 'Buhlmann Ceiling',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#ef5350', width: 1.5, dash: 'dash' },
        hovertemplate: 'Ceiling: %{customdata:.1f} m<extra>Buhlmann</extra>',
        customdata: buhlmannCeilings,
      });
    }

    // Slab ceiling line
    if (slabCeilings && slabCeilings.some(function (c) { return c > 0; })) {
      var sCeilTimes = [];
      var sStep = times.length / slabCeilings.length;
      for (var j = 0; j < slabCeilings.length; j++) {
        sCeilTimes.push(times[Math.min(Math.floor(j * sStep), times.length - 1)]);
      }
      traces.push({
        x: sCeilTimes,
        y: slabCeilings.map(function (c) { return -c; }),
        name: 'Slab Ceiling',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#fdd835', width: 1.5, dash: 'dot' },
        hovertemplate: 'Ceiling: %{customdata:.1f} m<extra>Slab</extra>',
        customdata: slabCeilings,
      });
    }

    var maxDepth = Math.max.apply(null, depths);
    var maxTime = Math.max.apply(null, times);

    var mobile = _isMobile();
    var layout = {
      paper_bgcolor: DARK_LAYOUT.paper_bgcolor,
      plot_bgcolor: DARK_LAYOUT.plot_bgcolor,
      font: DARK_LAYOUT.font,
      margin: _mobileMargin({ t: 30, r: 20, b: 50, l: 60 }),
      title: { text: 'Dive Profile', font: { size: mobile ? 12 : 14 } },
      xaxis: {
        title: mobile ? null : 'Time (min)',
        gridcolor: '#2a2a4a',
        zerolinecolor: '#2a2a4a',
        range: [0, maxTime * 1.05],
        tickfont: { size: mobile ? 9 : 12 },
      },
      yaxis: {
        title: mobile ? null : 'Depth (m)',
        gridcolor: '#2a2a4a',
        zerolinecolor: '#2a2a4a',
        range: [-(maxDepth * 1.15), 2],
        tickvals: _generateDepthTicks(maxDepth),
        ticktext: _generateDepthTicks(maxDepth).map(function (v) { return Math.abs(v).toString(); }),
        tickfont: { size: mobile ? 9 : 12 },
      },
      legend: _mobileLegend({
        x: 1,
        xanchor: 'right',
        y: 1,
        bgcolor: 'rgba(26, 26, 46, 0.8)',
        font: { size: 11 },
      }),
      showlegend: true,
    };

    Plotly.newPlot(elementId, traces, layout, PLOTLY_CONFIG);
  }

  function renderBuhlmannTissueChart(tissueN2, mValues) {
    var elementId = 'buhlmann-tissue-chart';
    Plotly.purge(elementId);

    var compartments = [];
    for (var i = 0; i < tissueN2.length; i++) {
      compartments.push('C' + (i + 1));
    }

    var tissueTrace = {
      x: compartments,
      y: tissueN2,
      name: 'Tissue N2',
      type: 'bar',
      marker: { color: '#4fc3f7' },
      hovertemplate: '%{x}<br>N2: %{y:.3f} bar<extra></extra>',
    };

    var mValueTrace = {
      x: compartments,
      y: mValues,
      name: 'M-value (GF)',
      type: 'scatter',
      mode: 'markers+lines',
      marker: { color: '#ef5350', size: 6 },
      line: { color: '#ef5350', width: 1.5 },
      hovertemplate: '%{x}<br>M-value: %{y:.3f} bar<extra></extra>',
    };

    var mobile = _isMobile();
    var layout = {
      paper_bgcolor: DARK_LAYOUT.paper_bgcolor,
      plot_bgcolor: DARK_LAYOUT.plot_bgcolor,
      font: DARK_LAYOUT.font,
      margin: _mobileMargin({ t: 30, r: 20, b: 40, l: 50 }),
      title: { text: 'Tissue Loading vs M-values', font: { size: mobile ? 10 : 12 } },
      xaxis: {
        gridcolor: '#2a2a4a',
        tickfont: { size: mobile ? 7 : 9 },
      },
      yaxis: {
        title: mobile ? null : 'Pressure (bar)',
        gridcolor: '#2a2a4a',
        zerolinecolor: '#2a2a4a',
        tickfont: { size: mobile ? 9 : 12 },
      },
      legend: _mobileLegend({
        x: 1,
        xanchor: 'right',
        y: 1,
        bgcolor: 'rgba(26, 26, 46, 0.8)',
        font: { size: 10 },
      }),
      barmode: 'group',
      showlegend: true,
    };

    Plotly.newPlot(elementId, [tissueTrace, mValueTrace], layout, PLOTLY_CONFIG);
  }

  function renderSlabCompartmentChart(compartments) {
    var elementId = 'slab-compartment-chart';
    Plotly.purge(elementId);

    var names = compartments.map(function (c) { return c.name; });

    // Gradient as fraction of g_crit
    var gradientFractions = compartments.map(function (c) {
      return c.gCrit > 0 ? c.gradient / c.gCrit : 0;
    });

    // Risk (excess gas / v_crit)
    var riskFractions = compartments.map(function (c) {
      return c.vCrit > 0 ? c.excessGas / c.vCrit : 0;
    });

    var gradientTrace = {
      x: names,
      y: gradientFractions,
      name: 'Gradient / g_crit',
      type: 'bar',
      marker: { color: '#fdd835' },
      hovertemplate: '%{x}<br>Gradient ratio: %{y:.3f}<extra></extra>',
    };

    var riskTrace = {
      x: names,
      y: riskFractions,
      name: 'Excess Gas / v_crit',
      type: 'bar',
      marker: { color: '#4fc3f7' },
      hovertemplate: '%{x}<br>Risk ratio: %{y:.3f}<extra></extra>',
    };

    // Critical threshold line at 1.0
    var thresholdTrace = {
      x: names,
      y: names.map(function () { return 1.0; }),
      name: 'Limit (1.0)',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#ef5350', width: 1.5, dash: 'dash' },
      hoverinfo: 'skip',
    };

    var mobile = _isMobile();
    var layout = {
      paper_bgcolor: DARK_LAYOUT.paper_bgcolor,
      plot_bgcolor: DARK_LAYOUT.plot_bgcolor,
      font: DARK_LAYOUT.font,
      margin: _mobileMargin({ t: 30, r: 20, b: 40, l: 50 }),
      title: { text: 'Compartment Loading', font: { size: mobile ? 10 : 12 } },
      xaxis: {
        gridcolor: '#2a2a4a',
        tickfont: { size: mobile ? 9 : 12 },
      },
      yaxis: {
        title: mobile ? null : 'Fraction of Limit',
        gridcolor: '#2a2a4a',
        zerolinecolor: '#2a2a4a',
        tickfont: { size: mobile ? 9 : 12 },
      },
      legend: _mobileLegend({
        x: 1,
        xanchor: 'right',
        y: 1,
        bgcolor: 'rgba(26, 26, 46, 0.8)',
        font: { size: 10 },
      }),
      barmode: 'group',
      showlegend: true,
    };

    Plotly.newPlot(elementId, [gradientTrace, riskTrace, thresholdTrace], layout, PLOTLY_CONFIG);
  }

  // Color assignments per compartment
  var COMPARTMENT_COLORS = {
    Spine:  { r: 79,  g: 195, b: 247 },  // cyan
    Muscle: { r: 253, g: 216, b: 53  },  // yellow
    Joints: { r: 239, g: 83,  b: 80  },  // red
  };

  function renderSlabGradientChart(compartments, ppN2Surface) {
    var canvas = document.getElementById('slab-gradient-chart');
    if (!canvas) return;

    var container = canvas.parentElement;
    var dpr = window.devicePixelRatio || 1;

    // Layout constants — scale for mobile
    var canvasWidth = container.clientWidth;
    var mobile = canvasWidth < 500;
    var barHeight = mobile ? Math.max(30, Math.round(canvasWidth * 0.09)) : 50;
    var barGap = mobile ? 8 : 12;
    var labelWidth = mobile ? Math.max(50, Math.round(canvasWidth * 0.15)) : 70;
    var rightPad = mobile ? 10 : 20;
    var topPad = mobile ? 22 : 30;
    var bottomPad = mobile ? 28 : 35;
    var legendHeight = 20;

    var numBars = compartments.length;
    var canvasHeight = topPad + numBars * barHeight + (numBars - 1) * barGap + bottomPad + legendHeight;

    // Set canvas size (CSS vs physical for retina)
    canvas.style.width = canvasWidth + 'px';
    canvas.style.height = canvasHeight + 'px';
    canvas.width = canvasWidth * dpr;
    canvas.height = canvasHeight * dpr;

    var ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    // Clear
    ctx.fillStyle = '#0f0f1a';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // Compute global min/max ppN2 across all slices for consistent mapping
    var globalMin = Infinity;
    var globalMax = -Infinity;
    for (var c = 0; c < compartments.length; c++) {
      var slices = compartments[c].slices;
      if (!slices) continue;
      for (var s = 0; s < slices.length; s++) {
        if (slices[s] < globalMin) globalMin = slices[s];
        if (slices[s] > globalMax) globalMax = slices[s];
      }
    }
    // Include surface ppN2 in range for context
    if (ppN2Surface < globalMin) globalMin = ppN2Surface;
    var range = globalMax - globalMin;
    if (range < 0.001) range = 0.001; // prevent division by zero

    var barLeft = labelWidth;
    var barWidth = canvasWidth - labelWidth - rightPad;

    // Draw each compartment bar
    for (var i = 0; i < compartments.length; i++) {
      var comp = compartments[i];
      var sliceData = comp.slices;
      if (!sliceData || sliceData.length === 0) continue;

      var color = COMPARTMENT_COLORS[comp.name] || { r: 200, g: 200, b: 200 };
      var y = topPad + i * (barHeight + barGap);
      var numSlices = sliceData.length;

      // Draw gradient bar pixel-by-pixel (column-wise)
      for (var px = 0; px < barWidth; px++) {
        // Map pixel position to fractional slice index
        var slicePos = (px / barWidth) * (numSlices - 1);
        var idx = Math.floor(slicePos);
        var frac = slicePos - idx;
        var nextIdx = Math.min(idx + 1, numSlices - 1);

        // Linear interpolation between adjacent slices
        var ppN2 = sliceData[idx] * (1 - frac) + sliceData[nextIdx] * frac;

        // Map to alpha (0.05 min so the bar outline is visible)
        var alpha = 0.05 + 0.95 * (ppN2 - globalMin) / range;

        ctx.fillStyle = 'rgba(' + color.r + ',' + color.g + ',' + color.b + ',' + alpha.toFixed(3) + ')';
        ctx.fillRect(barLeft + px, y, 1, barHeight);
      }

      // Bar outline
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
      ctx.lineWidth = 1;
      ctx.strokeRect(barLeft, y, barWidth, barHeight);

      // Surface ppN2 marker (vertical dashed line at equilibrium level)
      var eqAlpha = (ppN2Surface - globalMin) / range;
      var eqX = barLeft + eqAlpha * barWidth;
      if (eqX >= barLeft && eqX <= barLeft + barWidth) {
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(eqX, y);
        ctx.lineTo(eqX, y + barHeight);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Compartment label
      var labelFont = mobile ? '10px' : '12px';
      var subFont = mobile ? '8px' : '9px';
      ctx.fillStyle = 'rgba(' + color.r + ',' + color.g + ',' + color.b + ', 0.9)';
      ctx.font = labelFont + ' -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(comp.name, labelWidth - 8, y + barHeight / 2);

      // D value (diffusion coefficient)
      ctx.fillStyle = 'rgba(200, 200, 220, 0.5)';
      ctx.font = subFont + ' -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
      ctx.fillText('D=' + comp.D, labelWidth - 8, y + barHeight / 2 + (mobile ? 11 : 13));
    }

    // Axis labels
    var axisFont = mobile ? '9px' : '11px';
    var titleFont = mobile ? '10px' : '12px';
    ctx.fillStyle = 'rgba(200, 200, 220, 0.7)';
    ctx.font = axisFont + ' -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    // Title
    ctx.fillStyle = 'rgba(224, 224, 224, 0.9)';
    ctx.font = titleFont + ' -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(mobile ? 'ppN\u2082 Across Tissue Slabs' : 'ppN\u2082 Concentration Across Tissue Slabs', barLeft + barWidth / 2, 5);

    // "Blood" and "Core" axis labels below the bars
    var axisY = topPad + numBars * (barHeight + barGap) - barGap + 8;
    ctx.fillStyle = 'rgba(200, 200, 220, 0.7)';
    ctx.font = (mobile ? '9px' : '10px') + ' -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Blood', barLeft, axisY);
    ctx.textAlign = 'right';
    ctx.fillText('Core', barLeft + barWidth, axisY);

    // Arrow line between labels
    ctx.strokeStyle = 'rgba(200, 200, 220, 0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(barLeft + 35, axisY + 5);
    ctx.lineTo(barLeft + barWidth - 28, axisY + 5);
    ctx.stroke();
    // Arrowhead
    ctx.beginPath();
    ctx.moveTo(barLeft + barWidth - 28, axisY + 5);
    ctx.lineTo(barLeft + barWidth - 34, axisY + 2);
    ctx.lineTo(barLeft + barWidth - 34, axisY + 8);
    ctx.closePath();
    ctx.fillStyle = 'rgba(200, 200, 220, 0.3)';
    ctx.fill();

    // Legend: ppN2 color scale
    var legendY = canvasHeight - legendHeight - 5;
    var legendLeft = barLeft;
    var legendWidth = barWidth;
    var legendBarH = 8;

    // Draw gradient scale bar (grayscale)
    for (var lx = 0; lx < legendWidth; lx++) {
      var t = lx / legendWidth;
      var a = 0.05 + 0.95 * t;
      ctx.fillStyle = 'rgba(180, 180, 200,' + a.toFixed(3) + ')';
      ctx.fillRect(legendLeft + lx, legendY, 1, legendBarH);
    }

    // Legend labels
    ctx.fillStyle = 'rgba(200, 200, 220, 0.6)';
    ctx.font = (mobile ? '8px' : '9px') + ' -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(globalMin.toFixed(2) + ' bar', legendLeft, legendY + legendBarH + 2);
    ctx.textAlign = 'right';
    ctx.fillText(globalMax.toFixed(2) + ' bar', legendLeft + legendWidth, legendY + legendBarH + 2);
    ctx.textAlign = 'center';
    ctx.fillText('ppN\u2082', legendLeft + legendWidth / 2, legendY + legendBarH + 2);
  }

  function _generateDepthTicks(maxDepth) {
    var interval = 10;
    if (maxDepth > 100) interval = 20;
    else if (maxDepth > 50) interval = 10;
    else interval = 5;

    var ticks = [0];
    for (var d = interval; d <= maxDepth * 1.1; d += interval) {
      ticks.push(-d);
    }
    return ticks;
  }

  return {
    renderProfileChart: renderProfileChart,
    renderBuhlmannTissueChart: renderBuhlmannTissueChart,
    renderSlabCompartmentChart: renderSlabCompartmentChart,
    renderSlabGradientChart: renderSlabGradientChart,
  };
})();
