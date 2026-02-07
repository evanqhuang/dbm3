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

    var layout = {
      paper_bgcolor: DARK_LAYOUT.paper_bgcolor,
      plot_bgcolor: DARK_LAYOUT.plot_bgcolor,
      font: DARK_LAYOUT.font,
      margin: { t: 30, r: 20, b: 50, l: 60 },
      title: { text: 'Dive Profile', font: { size: 14 } },
      xaxis: {
        title: 'Time (min)',
        gridcolor: '#2a2a4a',
        zerolinecolor: '#2a2a4a',
        range: [0, maxTime * 1.05],
      },
      yaxis: {
        title: 'Depth (m)',
        gridcolor: '#2a2a4a',
        zerolinecolor: '#2a2a4a',
        range: [-(maxDepth * 1.15), 2],
        tickvals: _generateDepthTicks(maxDepth),
        ticktext: _generateDepthTicks(maxDepth).map(function (v) { return Math.abs(v).toString(); }),
      },
      legend: {
        x: 1,
        xanchor: 'right',
        y: 1,
        bgcolor: 'rgba(26, 26, 46, 0.8)',
        font: { size: 11 },
      },
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

    var layout = {
      paper_bgcolor: DARK_LAYOUT.paper_bgcolor,
      plot_bgcolor: DARK_LAYOUT.plot_bgcolor,
      font: DARK_LAYOUT.font,
      margin: { t: 30, r: 20, b: 40, l: 50 },
      title: { text: 'Tissue Loading vs M-values', font: { size: 12 } },
      xaxis: {
        gridcolor: '#2a2a4a',
        tickfont: { size: 9 },
      },
      yaxis: {
        title: 'Pressure (bar)',
        gridcolor: '#2a2a4a',
        zerolinecolor: '#2a2a4a',
      },
      legend: {
        x: 1,
        xanchor: 'right',
        y: 1,
        bgcolor: 'rgba(26, 26, 46, 0.8)',
        font: { size: 10 },
      },
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

    var layout = {
      paper_bgcolor: DARK_LAYOUT.paper_bgcolor,
      plot_bgcolor: DARK_LAYOUT.plot_bgcolor,
      font: DARK_LAYOUT.font,
      margin: { t: 30, r: 20, b: 40, l: 50 },
      title: { text: 'Compartment Loading', font: { size: 12 } },
      xaxis: {
        gridcolor: '#2a2a4a',
      },
      yaxis: {
        title: 'Fraction of Limit',
        gridcolor: '#2a2a4a',
        zerolinecolor: '#2a2a4a',
      },
      legend: {
        x: 1,
        xanchor: 'right',
        y: 1,
        bgcolor: 'rgba(26, 26, 46, 0.8)',
        font: { size: 10 },
      },
      barmode: 'group',
      showlegend: true,
    };

    Plotly.newPlot(elementId, [gradientTrace, riskTrace, thresholdTrace], layout, PLOTLY_CONFIG);
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
  };
})();
