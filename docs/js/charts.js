// charts.js — Plotly.js chart rendering for DBM-3 dashboard

const DARK_LAYOUT = {
  paper_bgcolor: '#1a1a2e',
  plot_bgcolor: '#0f0f1a',
  font: { color: '#e0e0e0', family: '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif' },
  margin: { t: 40, r: 20, b: 50, l: 60 },
  xaxis: { gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
  yaxis: { gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
};

function renderSummaryCards(gfSetting) {
  const container = document.getElementById('summary-cards');
  if (!container) return;

  const summary = gfSetting.summary;

  const cards = [
    {
      label: 'Risk Correlation',
      value: cardFormatCorrelation(summary.risk_correlation),
      class: cardClassCorrelation(summary.risk_correlation)
    },
    {
      label: 'NDL Correlation',
      value: cardFormatCorrelation(summary.ndl_correlation),
      class: cardClassCorrelation(summary.ndl_correlation)
    },
    {
      label: 'Ceiling Correlation',
      value: cardFormatCorrelation(summary.ceiling_correlation),
      class: cardClassCorrelation(summary.ceiling_correlation)
    },
    {
      label: 'Deco Agreement',
      value: cardFormatPercentage(summary.deco_agreement_pct),
      class: cardClassAgreement(summary.deco_agreement_pct)
    },
    {
      label: 'Total Profiles',
      value: summary.total_profiles.toString(),
      class: ''
    },
    {
      label: 'Mean Delta Risk',
      value: cardFormatDelta(summary.mean_delta_risk),
      class: cardClassDelta(summary.mean_delta_risk)
    }
  ];

  container.innerHTML = cards.map(card => `
    <div class="metric-card">
      <div class="card-label">${card.label}</div>
      <div class="card-value ${card.class}">${card.value}</div>
    </div>
  `).join('');
}

function cardFormatCorrelation(value) {
  if (value === undefined || value === null) return 'N/A';
  return `r=${value.toFixed(3)}`;
}

function cardFormatPercentage(value) {
  if (value === undefined || value === null) return 'N/A';
  return `${value.toFixed(1)}%`;
}

function cardFormatDelta(value) {
  if (value === undefined || value === null) return 'N/A';
  const sign = value >= 0 ? '+' : '';
  return `${sign}${value.toFixed(3)}`;
}

function cardClassCorrelation(value) {
  if (value === undefined || value === null) return 'cell-muted';
  if (value > 0.9) return 'good';
  if (value > 0.7) return 'warn';
  return 'bad';
}

function cardClassAgreement(value) {
  if (value === undefined || value === null) return 'cell-muted';
  if (value > 90) return 'good';
  if (value > 70) return 'warn';
  return 'bad';
}

function cardClassDelta(value) {
  if (value === undefined || value === null) return 'cell-muted';
  const abs = Math.abs(value);
  if (abs < 0.1) return 'good';
  if (abs < 0.3) return 'warn';
  return 'bad';
}

function renderGFComparisonTable(dataset, activeGFIndex) {
  const container = document.getElementById('gf-comparison-table-container');
  if (!container) return;

  const rows = dataset.gf_settings.map((gf, index) => {
    const summary = gf.summary;
    const isActive = index === activeGFIndex;

    return `
      <tr class="${isActive ? 'active-row' : ''}">
        <td>${gf.label}</td>
        <td>${formatTableCorrelation(summary.risk_correlation)}</td>
        <td>${formatTableCorrelation(summary.ndl_correlation)}</td>
        <td>${formatTableCorrelation(summary.ceiling_correlation)}</td>
        <td>${formatTableNumber(summary.deco_agreement_pct, 1)}</td>
        <td>${formatTableNumber(summary.mean_delta_risk, 3)}</td>
        <td>${formatTableNumber(summary.mean_delta_ndl, 3)}</td>
      </tr>
    `;
  }).join('');

  container.innerHTML = `
    <table class="gf-table">
      <thead>
        <tr>
          <th>GF</th>
          <th>Risk r</th>
          <th>NDL r</th>
          <th>Ceiling r</th>
          <th>Deco %</th>
          <th>Mean ΔRisk</th>
          <th>Mean ΔNDL</th>
        </tr>
      </thead>
      <tbody>
        ${rows}
      </tbody>
    </table>
  `;
}

function formatTableCorrelation(value) {
  if (value === undefined || value === null) {
    return 'N/A';
  }
  return value.toFixed(3);
}

function formatTableNumber(value, decimals) {
  if (value === undefined || value === null) {
    return 'N/A';
  }
  return value.toFixed(decimals);
}

function renderGFComparisonChart(dataset) {
  const elementId = 'gf-comparison-chart';
  Plotly.purge(elementId);

  const labels = dataset.gf_settings.map(gf => gf.label);
  const riskCorr = dataset.gf_settings.map(gf => gf.summary.risk_correlation || 0);
  const ndlCorr = dataset.gf_settings.map(gf => gf.summary.ndl_correlation || 0);
  const ceilingCorr = dataset.gf_settings.map(gf => gf.summary.ceiling_correlation || 0);

  const traces = [
    {
      x: labels,
      y: riskCorr,
      name: 'Risk Correlation',
      type: 'bar',
      marker: { color: '#4fc3f7' }
    },
    {
      x: labels,
      y: ndlCorr,
      name: 'NDL Correlation',
      type: 'bar',
      marker: { color: '#66bb6a' }
    },
    {
      x: labels,
      y: ceilingCorr,
      name: 'Ceiling Correlation',
      type: 'bar',
      marker: { color: '#fdd835' }
    }
  ];

  const layout = {
    ...DARK_LAYOUT,
    title: 'Correlation Metrics Across GF Settings',
    barmode: 'group',
    yaxis: {
      ...DARK_LAYOUT.yaxis,
      title: 'Correlation (r)',
      range: [0, 1]
    },
    xaxis: {
      ...DARK_LAYOUT.xaxis,
      title: 'GF Setting'
    },
    legend: {
      x: 1,
      xanchor: 'right',
      y: 1
    }
  };

  Plotly.newPlot(elementId, traces, layout, { responsive: true });
}

function renderHeatmaps(gfSetting, metric) {
  const matrices = gfSetting.matrices;
  if (!matrices) return;

  const buhlmannMatrix = matrices[`buhlmann_${metric}`];
  const slabMatrix = matrices[`slab_${metric}`];
  const deltaMatrix = matrices[`delta_${metric}`];

  if (!buhlmannMatrix || !slabMatrix || !deltaMatrix) return;

  const buhlmannFlat = buhlmannMatrix.data.flat();
  const slabFlat = slabMatrix.data.flat();
  const deltaFlat = deltaMatrix.data.flat();

  const sharedMin = Math.min(...buhlmannFlat, ...slabFlat);
  const sharedMax = Math.max(...buhlmannFlat, ...slabFlat);

  const deltaAbsMax = Math.max(Math.abs(Math.min(...deltaFlat)), Math.abs(Math.max(...deltaFlat)));

  renderSingleHeatmap(
    'heatmap-buhlmann',
    buhlmannMatrix,
    `Buhlmann ${capitalizeMetric(metric)}`,
    'YlOrRd',
    [sharedMin, sharedMax],
    metric
  );

  renderSingleHeatmap(
    'heatmap-slab',
    slabMatrix,
    `Slab ${capitalizeMetric(metric)}`,
    'YlOrRd',
    [sharedMin, sharedMax],
    metric
  );

  renderSingleHeatmap(
    'heatmap-delta',
    deltaMatrix,
    `Delta ${capitalizeMetric(metric)} (Slab - Buhlmann)`,
    [[0, '#1565c0'], [0.5, '#1a1a2e'], [1, '#c62828']],
    [-deltaAbsMax, deltaAbsMax],
    metric
  );
}

function renderSingleHeatmap(elementId, matrix, title, colorscale, zrange, metric) {
  Plotly.purge(elementId);

  const trace = {
    z: matrix.data,
    x: matrix.depths,
    y: matrix.times,
    type: 'heatmap',
    colorscale: colorscale,
    zmin: zrange[0],
    zmax: zrange[1],
    hovertemplate: `Depth: %{x}m<br>Time: %{y}min<br>Value: %{z:.3f}<extra></extra>`
  };

  const layout = {
    ...DARK_LAYOUT,
    title: title,
    xaxis: {
      ...DARK_LAYOUT.xaxis,
      title: 'Depth (m)'
    },
    yaxis: {
      ...DARK_LAYOUT.yaxis,
      title: 'Time (min)'
    }
  };

  Plotly.newPlot(elementId, [trace], layout, { responsive: true });
}

function capitalizeMetric(metric) {
  const map = {
    risk: 'Risk',
    ndl: 'NDL',
    ceiling: 'Ceiling'
  };
  return map[metric] || metric;
}

function renderScatter(gfSetting, metric, isReal) {
  const elementId = 'scatter-chart';
  Plotly.purge(elementId);

  const profiles = gfSetting.profiles;
  if (!profiles || profiles.length === 0) return;

  const xField = metric === 'ceiling' ? 'buhlmann_ceiling_m' : `buhlmann_${metric}`;
  const yField = metric === 'ceiling' ? 'slab_ceiling_m' : `slab_${metric}`;
  const depthField = isReal ? 'max_depth' : 'depth_m';
  const timeField = isReal ? 'bottom_time' : 'bottom_time_min';

  const xVals = [];
  const yVals = [];
  const depths = [];
  const texts = [];

  profiles.forEach(p => {
    const xVal = p[xField];
    const yVal = p[yField];
    if (xVal !== undefined && yVal !== undefined && xVal !== null && yVal !== null) {
      xVals.push(xVal);
      yVals.push(yVal);
      depths.push(p[depthField]);
      texts.push(
        `${p.profile_name}<br>` +
        `Depth: ${p[depthField]}m, Time: ${p[timeField]}min<br>` +
        `Buhlmann: ${xVal.toFixed(3)}<br>` +
        `Slab: ${yVal.toFixed(3)}`
      );
    }
  });

  if (xVals.length === 0) return;

  const r = pearsonR(xVals, yVals);

  const dataMin = Math.min(...xVals, ...yVals);
  const dataMax = Math.max(...xVals, ...yVals);

  const scatterTrace = {
    x: xVals,
    y: yVals,
    mode: 'markers',
    type: 'scatter',
    marker: {
      size: 4,
      color: depths,
      colorscale: 'Viridis',
      opacity: 0.6,
      colorbar: {
        title: 'Depth (m)',
        tickfont: { color: '#e0e0e0' }
      }
    },
    text: texts,
    hovertemplate: '%{text}<extra></extra>'
  };

  const identityTrace = {
    x: [dataMin, dataMax],
    y: [dataMin, dataMax],
    mode: 'lines',
    type: 'scatter',
    line: {
      color: '#666',
      dash: 'dash',
      width: 1
    },
    hoverinfo: 'skip',
    showlegend: false
  };

  const metricLabel = capitalizeMetric(metric);

  const layout = {
    ...DARK_LAYOUT,
    title: `${metricLabel} — Slab vs Buhlmann (r = ${r.toFixed(3)})`,
    xaxis: {
      ...DARK_LAYOUT.xaxis,
      title: `Buhlmann ${metricLabel}`
    },
    yaxis: {
      ...DARK_LAYOUT.yaxis,
      title: `Slab ${metricLabel}`
    }
  };

  Plotly.newPlot(elementId, [scatterTrace, identityTrace], layout, { responsive: true });
}

function pearsonR(x, y) {
  const n = x.length;
  if (n === 0) return 0;

  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;

  let num = 0;
  let dx2 = 0;
  let dy2 = 0;

  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx;
    const dy = y[i] - my;
    num += dx * dy;
    dx2 += dx * dx;
    dy2 += dy * dy;
  }

  if (dx2 === 0 || dy2 === 0) return 0;

  return num / Math.sqrt(dx2 * dy2);
}

function renderScatterStats(gfSetting, metric, isReal) {
  const container = document.getElementById('scatter-stats');
  if (!container) return;

  const profiles = gfSetting.profiles;
  if (!profiles || profiles.length === 0) {
    container.innerHTML = '<div class="stat-item">No data</div>';
    return;
  }

  const xField = metric === 'ceiling' ? 'buhlmann_ceiling_m' : `buhlmann_${metric}`;
  const yField = metric === 'ceiling' ? 'slab_ceiling_m' : `slab_${metric}`;

  const xVals = [];
  const yVals = [];
  let slabConservativeCount = 0;
  let buhlmannConservativeCount = 0;

  profiles.forEach(p => {
    const xVal = p[xField];
    const yVal = p[yField];
    if (xVal !== undefined && yVal !== undefined && xVal !== null && yVal !== null) {
      xVals.push(xVal);
      yVals.push(yVal);

      if (metric === 'ndl') {
        if (yVal < xVal) slabConservativeCount++;
        if (xVal < yVal) buhlmannConservativeCount++;
      } else {
        if (yVal > xVal) slabConservativeCount++;
        if (xVal > yVal) buhlmannConservativeCount++;
      }
    }
  });

  if (xVals.length === 0) {
    container.innerHTML = '<div class="stat-item">No valid data</div>';
    return;
  }

  const r = pearsonR(xVals, yVals);

  let sumAbsDiff = 0;
  for (let i = 0; i < xVals.length; i++) {
    sumAbsDiff += Math.abs(yVals[i] - xVals[i]);
  }
  const meanAbsDiff = sumAbsDiff / xVals.length;

  const stats = [
    {
      label: 'Correlation (r)',
      value: r.toFixed(3)
    },
    {
      label: 'Mean Absolute Difference',
      value: meanAbsDiff.toFixed(3)
    },
    {
      label: 'Slab More Conservative',
      value: slabConservativeCount.toString()
    },
    {
      label: 'Buhlmann More Conservative',
      value: buhlmannConservativeCount.toString()
    }
  ];

  container.innerHTML = stats.map(stat => `
    <div class="stat-item">
      <div class="stat-label">${stat.label}</div>
      <div class="stat-value">${stat.value}</div>
    </div>
  `).join('');
}
