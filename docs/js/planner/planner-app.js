// planner-app.js — Main controller for DBM-3 Dive Planner

(function () {
  'use strict';

  var pyodide = null;

  // Python files to load into Pyodide virtual filesystem
  var PY_FILES = [
    { url: 'py/planner_bridge.py', path: '/home/pyodide/planner_bridge.py' },
    { url: 'py/backtest/__init__.py', path: '/home/pyodide/backtest/__init__.py' },
    { url: 'py/backtest/profile_generator.py', path: '/home/pyodide/backtest/profile_generator.py' },
    { url: 'py/backtest/buhlmann_constants.py', path: '/home/pyodide/backtest/buhlmann_constants.py' },
    { url: 'py/backtest/buhlmann_engine.py', path: '/home/pyodide/backtest/buhlmann_engine.py' },
    { url: 'py/backtest/slab_model.py', path: '/home/pyodide/backtest/slab_model.py' },
  ];

  function setLoadingText(text) {
    var el = document.getElementById('loading-text');
    if (el) el.textContent = text;
  }

  function hideLoading() {
    var el = document.getElementById('loading-overlay');
    if (el) el.classList.add('hidden');
  }

  function showLoadingError(message) {
    setLoadingText('Error: ' + message);
    var spinner = document.querySelector('.loading-overlay .spinner');
    if (spinner) spinner.style.display = 'none';
  }

  async function fetchTextFile(url) {
    var response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch ' + url + ': ' + response.status);
    return await response.text();
  }

  async function initPyodide() {
    try {
      // Step 1: Load Pyodide runtime
      setLoadingText('Loading Python runtime...');
      pyodide = await loadPyodide();

      // Step 2: Load numpy
      setLoadingText('Loading numpy...');
      await pyodide.loadPackage('numpy');

      // Step 3: Create backtest directory in virtual FS
      setLoadingText('Loading model files...');
      pyodide.FS.mkdir('/home/pyodide/backtest');

      // Step 4: Fetch and write Python files
      for (var i = 0; i < PY_FILES.length; i++) {
        var file = PY_FILES[i];
        var content = await fetchTextFile(file.url);
        pyodide.FS.writeFile(file.path, content);
      }

      // Step 5: Mock yaml module (slab_model.py imports yaml at top level)
      // and import the bridge
      setLoadingText('Initializing models...');
      await pyodide.runPythonAsync(`
import types, sys
yaml = types.ModuleType('yaml')
yaml.safe_load = lambda x: {}
sys.modules['yaml'] = yaml

sys.path.insert(0, '/home/pyodide')
import planner_bridge
`);

      // Step 6: Ready
      setLoadingText('Ready!');
      hideLoading();
      document.getElementById('compute-btn').disabled = false;

    } catch (err) {
      console.error('Pyodide init failed:', err);
      showLoadingError(err.message || 'Failed to initialize Python engine');
    }
  }

  async function compute() {
    PlannerUI.hideError();

    var params = PlannerUI.getParams();
    var error = PlannerUI.validateParams(params);
    if (error) {
      PlannerUI.showError(error);
      return;
    }

    PlannerUI.setComputing(true);

    try {
      var paramsJson = JSON.stringify(params);
      var resultJson = await pyodide.runPythonAsync(
        'planner_bridge.plan_dive(\'' + paramsJson.replace(/\\/g, '\\\\').replace(/'/g, "\\'") + '\')'
      );

      var result = JSON.parse(resultJson);

      if (result.error) {
        PlannerUI.showError(result.error);
        PlannerUI.setComputing(false);
        return;
      }

      // Render profile chart
      PlannerCharts.renderProfileChart(
        result.profile,
        result.buhlmann.ceilingsOverTime,
        result.slab.ceilingsOverTime
      );

      // Render summaries
      PlannerUI.renderBuhlmannSummary(result.buhlmann);
      PlannerUI.renderSlabSummary(result.slab);

      // Render tissue/compartment charts
      PlannerCharts.renderBuhlmannTissueChart(result.buhlmann.tissueN2, result.buhlmann.mValues);
      PlannerCharts.renderSlabCompartmentChart(result.slab.compartments);

      // Render deco schedules
      PlannerUI.renderDecoSchedule(result.buhlmann, result.slab);

      // Show results section
      PlannerUI.showResults();

    } catch (err) {
      console.error('Compute failed:', err);
      PlannerUI.showError('Computation failed: ' + (err.message || err));
    }

    PlannerUI.setComputing(false);
  }

  function bindEvents() {
    document.getElementById('compute-btn').addEventListener('click', compute);

    document.getElementById('add-level-btn').addEventListener('click', function () {
      PlannerUI.addLevel(15, 10);
    });
  }

  function init() {
    PlannerUI.init();
    bindEvents();
    initPyodide();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
