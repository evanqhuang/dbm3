/* DBM-3 Dashboard — App Controller */

(function () {
  'use strict';

  // State
  const state = {
    datasetIndex: 0,
    gfIndex: 0,
    activeTab: 'summary',
    rendered: {},
  };

  // DOM refs
  const els = {
    spinner: document.getElementById('loading-spinner'),
    datasetSelect: document.getElementById('dataset-select'),
    gfSelect: document.getElementById('gf-select'),
    tabs: document.querySelectorAll('.tab-btn'),
    panels: document.querySelectorAll('.tab-panel'),
    heatmapMetric: document.getElementById('heatmap-metric'),
    scatterMetric: document.getElementById('scatter-metric'),
    profileSearch: document.getElementById('profile-search'),
    profileSort: document.getElementById('profile-sort'),
    heatmapFallback: document.getElementById('heatmap-fallback'),
    heatmapPanels: document.getElementById('heatmap-panels'),
    scatterFallback: document.getElementById('scatter-fallback'),
    scatterChart: document.getElementById('scatter-chart'),
    scatterStats: document.getElementById('scatter-stats'),
  };

  // Helpers
  function getCurrentDataset() {
    return SWEEP_DATA.datasets[state.datasetIndex];
  }

  function getCurrentGF() {
    return getCurrentDataset().gf_settings[state.gfIndex];
  }

  function isReal() {
    return getCurrentDataset().source === 'real';
  }

  // Populate dataset selector
  function populateDatasetSelect() {
    els.datasetSelect.innerHTML = SWEEP_DATA.datasets
      .map(function (ds, i) {
        var label = ds.source.charAt(0).toUpperCase() + ds.source.slice(1);
        return '<option value="' + i + '">' + label + ' (' + ds.total_profiles + ' profiles)</option>';
      })
      .join('');
    els.datasetSelect.value = state.datasetIndex;
  }

  // Populate GF selector for current dataset
  function populateGFSelect() {
    var dataset = getCurrentDataset();
    els.gfSelect.innerHTML = dataset.gf_settings
      .map(function (gs, i) {
        return '<option value="' + i + '">GF ' + gs.gf_low + '/' + gs.gf_high + ' — ' + gs.label + '</option>';
      })
      .join('');
    els.gfSelect.value = state.gfIndex;
  }

  // Tab switching
  function switchTab(tabName) {
    state.activeTab = tabName;
    els.tabs.forEach(function (btn) {
      btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    els.panels.forEach(function (panel) {
      panel.classList.toggle('active', panel.id === 'tab-' + tabName);
    });
    renderActiveTab();
  }

  // Render the currently active tab
  function renderActiveTab() {
    var gf = getCurrentGF();
    var dataset = getCurrentDataset();
    var real = isReal();

    switch (state.activeTab) {
      case 'summary':
        renderSummaryCards(gf);
        renderGFComparisonTable(dataset, state.gfIndex);
        renderGFComparisonChart(dataset);
        break;

      case 'heatmaps':
        if (real || !gf.matrices) {
          els.heatmapFallback.style.display = '';
          els.heatmapPanels.style.display = 'none';
        } else {
          els.heatmapFallback.style.display = 'none';
          els.heatmapPanels.style.display = '';
          renderHeatmaps(gf, els.heatmapMetric.value);
        }
        break;

      case 'scatter':
        var metric = els.scatterMetric.value;
        if (real && metric === 'ceiling') {
          els.scatterFallback.style.display = '';
          els.scatterChart.style.display = 'none';
          els.scatterStats.style.display = 'none';
        } else {
          els.scatterFallback.style.display = 'none';
          els.scatterChart.style.display = '';
          els.scatterStats.style.display = '';
          renderScatter(gf, metric, real);
          renderScatterStats(gf, metric, real);
        }
        break;

      case 'profiles':
        var tableState = getTableState();
        renderProfileTable(gf, real, {
          searchTerm: tableState.searchTerm,
          sortKey: tableState.sortKey,
          sortDir: tableState.sortDir,
          page: tableState.page,
          pageSize: tableState.pageSize,
        });
        break;
    }
  }

  // Event: dataset change
  function onDatasetChange() {
    state.datasetIndex = parseInt(els.datasetSelect.value, 10);
    state.gfIndex = 0;
    populateGFSelect();
    renderActiveTab();
  }

  // Event: GF change
  function onGFChange() {
    state.gfIndex = parseInt(els.gfSelect.value, 10);
    renderActiveTab();
  }

  // Bind events
  function bindEvents() {
    els.datasetSelect.addEventListener('change', onDatasetChange);
    els.gfSelect.addEventListener('change', onGFChange);

    els.tabs.forEach(function (btn) {
      btn.addEventListener('click', function () {
        switchTab(btn.dataset.tab);
      });
    });

    els.heatmapMetric.addEventListener('change', function () {
      renderActiveTab();
    });

    els.scatterMetric.addEventListener('change', function () {
      renderActiveTab();
    });

    // Heatmap fallback button
    document.getElementById('heatmap-switch-btn').addEventListener('click', function () {
      // Find synthetic dataset index
      for (var i = 0; i < SWEEP_DATA.datasets.length; i++) {
        if (SWEEP_DATA.datasets[i].source === 'synthetic') {
          state.datasetIndex = i;
          state.gfIndex = 0;
          els.datasetSelect.value = i;
          populateGFSelect();
          renderActiveTab();
          break;
        }
      }
    });

    // Scatter fallback button
    document.getElementById('scatter-switch-btn').addEventListener('click', function () {
      for (var i = 0; i < SWEEP_DATA.datasets.length; i++) {
        if (SWEEP_DATA.datasets[i].source === 'synthetic') {
          state.datasetIndex = i;
          state.gfIndex = 0;
          els.datasetSelect.value = i;
          populateGFSelect();
          renderActiveTab();
          break;
        }
      }
    });

    // Profile search
    els.profileSearch.addEventListener('input', function () {
      var tableState = getTableState();
      tableState.searchTerm = els.profileSearch.value;
      tableState.page = 0;
      renderProfileTable(getCurrentGF(), isReal(), tableState);
    });

    // Profile sort dropdown
    els.profileSort.addEventListener('change', function () {
      var tableState = getTableState();
      tableState.sortKey = els.profileSort.value;
      tableState.page = 0;
      renderProfileTable(getCurrentGF(), isReal(), tableState);
    });

    // Table internal callbacks (sort clicks, pagination)
    setTableCallbacks(function (newState) {
      renderProfileTable(getCurrentGF(), isReal(), newState);
    });
  }

  // Initialize
  function init() {
    populateDatasetSelect();
    populateGFSelect();
    bindEvents();
    renderActiveTab();
    els.spinner.classList.add('hidden');
  }

  // Wait for DOM
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
