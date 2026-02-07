// planner-ui.js — DOM manipulation for DBM-3 Dive Planner

var PlannerUI = (function () {
  'use strict';

  var MAX_LEVELS = 10;
  var levelsContainer = null;
  var levelCount = 0;

  function init() {
    levelsContainer = document.getElementById('levels-container');
    addLevel(30, 20);  // Default first level
  }

  function addLevel(depth, duration) {
    if (levelCount >= MAX_LEVELS) return;
    levelCount++;

    var row = document.createElement('div');
    row.className = 'level-row';
    row.dataset.level = levelCount;

    row.innerHTML =
      '<span class="level-label">Level ' + levelCount + '</span>' +
      '<div class="control-group">' +
        '<label>Depth (m)</label>' +
        '<input type="number" class="level-depth" value="' + depth + '" min="1" max="300" step="1">' +
      '</div>' +
      '<div class="control-group">' +
        '<label>Duration (min)</label>' +
        '<input type="number" class="level-duration" value="' + duration + '" min="1" max="300" step="1">' +
      '</div>' +
      '<button class="btn-remove" title="Remove level">Remove</button>';

    row.querySelector('.btn-remove').addEventListener('click', function () {
      row.remove();
      levelCount--;
      renumberLevels();
    });

    levelsContainer.appendChild(row);
  }

  function renumberLevels() {
    var rows = levelsContainer.querySelectorAll('.level-row');
    for (var i = 0; i < rows.length; i++) {
      rows[i].querySelector('.level-label').textContent = 'Level ' + (i + 1);
      rows[i].dataset.level = i + 1;
    }
    levelCount = rows.length;
  }

  function getLevels() {
    var rows = levelsContainer.querySelectorAll('.level-row');
    var levels = [];
    for (var i = 0; i < rows.length; i++) {
      var depth = parseFloat(rows[i].querySelector('.level-depth').value);
      var duration = parseFloat(rows[i].querySelector('.level-duration').value);
      levels.push({ depth: depth, duration: duration });
    }
    return levels;
  }

  function getParams() {
    return {
      levels: getLevels(),
      fO2: parseFloat(document.getElementById('input-fo2').value),
      fHe: parseFloat(document.getElementById('input-fhe').value),
      gfLow: parseFloat(document.getElementById('input-gf-low').value) / 100.0,
      gfHigh: parseFloat(document.getElementById('input-gf-high').value) / 100.0,
      conservatism: parseFloat(document.getElementById('input-conservatism').value),
    };
  }

  function validateParams(params) {
    if (!params.levels || params.levels.length === 0) {
      return 'At least one dive level is required.';
    }
    for (var i = 0; i < params.levels.length; i++) {
      var level = params.levels[i];
      if (isNaN(level.depth) || level.depth <= 0 || level.depth > 300) {
        return 'Level ' + (i + 1) + ': depth must be between 1 and 300 meters.';
      }
      if (isNaN(level.duration) || level.duration <= 0 || level.duration > 300) {
        return 'Level ' + (i + 1) + ': duration must be between 1 and 300 minutes.';
      }
    }
    if (isNaN(params.fO2) || params.fO2 < 0.1 || params.fO2 > 1.0) {
      return 'fO2 must be between 0.10 and 1.00.';
    }
    if (isNaN(params.fHe) || params.fHe < 0.0 || params.fHe > 0.8) {
      return 'fHe must be between 0.00 and 0.80.';
    }
    if (params.fO2 + params.fHe > 1.0) {
      return 'fO2 + fHe cannot exceed 1.0.';
    }
    if (isNaN(params.gfLow) || params.gfLow <= 0 || params.gfLow > 1.0) {
      return 'GF Low must be between 1 and 100.';
    }
    if (isNaN(params.gfHigh) || params.gfHigh <= 0 || params.gfHigh > 1.0) {
      return 'GF High must be between 1 and 100.';
    }
    if (params.gfLow > params.gfHigh) {
      return 'GF Low must be less than or equal to GF High.';
    }
    if (isNaN(params.conservatism) || params.conservatism < 0.5 || params.conservatism > 1.5) {
      return 'Conservatism must be between 0.50 and 1.50.';
    }
    return null;  // valid
  }

  function showError(message) {
    var el = document.getElementById('error-display');
    el.textContent = message;
    el.style.display = '';
  }

  function hideError() {
    var el = document.getElementById('error-display');
    el.style.display = 'none';
  }

  function setComputing(isComputing) {
    var btn = document.getElementById('compute-btn');
    if (isComputing) {
      btn.textContent = 'Computing...';
      btn.disabled = true;
      document.body.classList.add('computing');
    } else {
      btn.textContent = 'Compute';
      btn.disabled = false;
      document.body.classList.remove('computing');
    }
  }

  function renderBuhlmannSummary(data) {
    var container = document.getElementById('buhlmann-summary');
    container.innerHTML = _buildMetrics([
      { label: 'Risk', value: data.risk.toFixed(3), cls: _riskClass(data.risk) },
      { label: 'NDL', value: data.ndl.toFixed(1) + ' min', cls: _ndlClass(data.ndl) },
      { label: 'Max Ceiling', value: data.maxCeiling.toFixed(1) + ' m', cls: data.maxCeiling > 0 ? 'warn' : 'good' },
      { label: 'Deco Required', value: data.requiresDeco ? 'Yes' : 'No', cls: data.requiresDeco ? 'warn' : 'good' },
    ]);
  }

  function renderSlabSummary(data) {
    var container = document.getElementById('slab-summary');
    container.innerHTML = _buildMetrics([
      { label: 'Risk', value: data.risk.toFixed(3), cls: _riskClass(data.risk) },
      { label: 'NDL', value: data.ndl.toFixed(1) + ' min', cls: _ndlClass(data.ndl) },
      { label: 'Ceiling', value: data.ceiling.toFixed(1) + ' m', cls: data.ceiling > 0 ? 'warn' : 'good' },
      { label: 'Deco Required', value: data.requiresDeco ? 'Yes' : 'No', cls: data.requiresDeco ? 'warn' : 'good' },
      { label: 'Critical', value: data.criticalCompartment, cls: '' },
    ]);
  }

  function renderDecoSchedule(buhlmannData, slabData) {
    var section = document.getElementById('deco-section');
    var showDeco = buhlmannData.requiresDeco || slabData.requiresDeco;
    section.style.display = showDeco ? '' : 'none';

    if (!showDeco) return;

    document.getElementById('buhlmann-deco').innerHTML = _buildDecoTable(buhlmannData.decoStops, buhlmannData.requiresDeco);
    document.getElementById('slab-deco').innerHTML = _buildDecoTable(slabData.decoStops, slabData.requiresDeco);
  }

  function showResults() {
    document.getElementById('results-section').style.display = '';
  }

  function _buildMetrics(metrics) {
    return metrics.map(function (m) {
      return '<div class="result-metric">' +
        '<div class="metric-label">' + m.label + '</div>' +
        '<div class="metric-value ' + m.cls + '">' + m.value + '</div>' +
        '</div>';
    }).join('');
  }

  function _buildDecoTable(stops, requiresDeco) {
    if (!requiresDeco || !stops || stops.length === 0) {
      return '<div class="no-deco-message">No decompression required</div>';
    }

    var totalTime = 0;
    var rows = stops.map(function (stop) {
      totalTime += stop.duration;
      return '<tr><td>' + stop.depth.toFixed(0) + ' m</td><td>' + stop.duration.toFixed(0) + ' min</td></tr>';
    }).join('');

    return '<table class="deco-table">' +
      '<thead><tr><th>Depth</th><th>Duration</th></tr></thead>' +
      '<tbody>' + rows +
      '<tr class="tts-row"><td>TTS</td><td>' + totalTime.toFixed(0) + ' min</td></tr>' +
      '</tbody></table>';
  }

  function _riskClass(risk) {
    if (risk < 0.7) return 'good';
    if (risk < 1.0) return 'warn';
    return 'bad';
  }

  function _ndlClass(ndl) {
    if (ndl > 10) return 'good';
    if (ndl > 0) return 'warn';
    return 'bad';
  }

  return {
    init: init,
    addLevel: addLevel,
    getParams: getParams,
    validateParams: validateParams,
    showError: showError,
    hideError: hideError,
    setComputing: setComputing,
    renderBuhlmannSummary: renderBuhlmannSummary,
    renderSlabSummary: renderSlabSummary,
    renderDecoSchedule: renderDecoSchedule,
    showResults: showResults,
  };
})();
