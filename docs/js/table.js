// table.js - Profile data table module for DBM-3 dashboard

// Module state
let _tableState = {
  searchTerm: '',
  sortKey: 'profile_name',
  sortDir: 'asc',
  page: 0,
  pageSize: 50,
};

let _onStateChange = null;

// Column definitions
const SYNTHETIC_COLS = [
  { key: 'profile_name', label: 'Name', align: 'left' },
  { key: 'depth_m', label: 'Depth', format: 'depth' },
  { key: 'bottom_time_min', label: 'Time', format: 'time' },
  { key: 'buhlmann_risk', label: 'B.Risk', format: 'risk' },
  { key: 'slab_risk', label: 'S.Risk', format: 'risk' },
  { key: 'delta_risk', label: 'ΔRisk', format: 'delta_risk' },
  { key: 'buhlmann_ndl', label: 'B.NDL', format: 'ndl' },
  { key: 'slab_ndl', label: 'S.NDL', format: 'ndl' },
  { key: 'delta_ndl', label: 'ΔNDL', format: 'delta_ndl' },
  { key: 'buhlmann_ceiling_m', label: 'B.Ceil', format: 'ceil' },
  { key: 'slab_ceiling_m', label: 'S.Ceil', format: 'ceil' },
  { key: 'delta_ceiling_m', label: 'ΔCeil', format: 'delta_ceil' },
  { key: 'slab_critical_compartment', label: 'Critical', format: 'text' },
];

const REAL_COLS = [
  { key: 'profile_name', label: 'Name', align: 'left' },
  { key: 'max_depth', label: 'Depth', format: 'depth' },
  { key: 'bottom_time', label: 'Time', format: 'time' },
  { key: 'buhlmann_risk', label: 'B.Risk', format: 'risk' },
  { key: 'slab_risk', label: 'S.Risk', format: 'risk' },
  { key: 'delta_risk', label: 'ΔRisk', format: 'delta_risk' },
  { key: 'buhlmann_ndl', label: 'B.NDL', format: 'ndl' },
  { key: 'slab_ndl', label: 'S.NDL', format: 'ndl' },
  { key: 'delta_ndl', label: 'ΔNDL', format: 'delta_ndl' },
];

// Public API
function setTableCallbacks(onStateChange) {
  _onStateChange = onStateChange;
}

function getTableState() {
  return { ..._tableState };
}

function renderProfileTable(gfSetting, isReal, options) {
  // Update internal state from options
  _tableState.searchTerm = options.searchTerm || '';
  _tableState.sortKey = options.sortKey || 'profile_name';
  _tableState.sortDir = options.sortDir || 'asc';
  _tableState.page = options.page || 0;
  _tableState.pageSize = options.pageSize || 50;

  const cols = isReal ? REAL_COLS : SYNTHETIC_COLS;
  const profiles = gfSetting.profiles || [];

  // Filter profiles by search term
  const filtered = filterProfiles(profiles, _tableState.searchTerm);

  // Sort profiles
  const sorted = sortProfiles(filtered, _tableState.sortKey, _tableState.sortDir);

  // Paginate
  const totalPages = Math.ceil(sorted.length / _tableState.pageSize);
  const start = _tableState.page * _tableState.pageSize;
  const end = start + _tableState.pageSize;
  const page = sorted.slice(start, end);

  // Render table
  renderTable(cols, page, _tableState.sortKey, _tableState.sortDir);

  // Render pagination
  renderPagination(sorted.length, _tableState.page, totalPages);
}

// Filter profiles by search term
function filterProfiles(profiles, searchTerm) {
  if (!searchTerm) {
    return profiles;
  }

  const term = searchTerm.toLowerCase();
  return profiles.filter(p => {
    const name = (p.profile_name || '').toLowerCase();
    return name.includes(term);
  });
}

// Sort profiles
function sortProfiles(profiles, sortKey, sortDir) {
  const sorted = [...profiles].sort((a, b) => {
    const aVal = a[sortKey];
    const bVal = b[sortKey];

    // Handle null/undefined
    if (aVal == null && bVal == null) return 0;
    if (aVal == null) return 1;
    if (bVal == null) return -1;

    // String comparison
    if (typeof aVal === 'string' && typeof bVal === 'string') {
      return aVal.localeCompare(bVal);
    }

    // Numeric comparison
    return aVal - bVal;
  });

  return sortDir === 'desc' ? sorted.reverse() : sorted;
}

// Render table HTML
function renderTable(cols, profiles, sortKey, sortDir) {
  const container = document.getElementById('profile-table-container');
  if (!container) return;

  if (profiles.length === 0) {
    container.innerHTML = '<div class="empty-state">No profiles match your search.</div>';
    return;
  }

  let html = '<table class="profile-table">';

  // Header row
  html += '<thead><tr>';
  for (const col of cols) {
    const isSorted = col.key === sortKey;
    const sortClass = isSorted ? `sort-${sortDir}` : '';
    const align = col.align || 'right';
    html += `<th class="col-${align} ${sortClass}" data-key="${col.key}">${col.label}</th>`;
  }
  html += '</tr></thead>';

  // Body rows
  html += '<tbody>';
  for (const profile of profiles) {
    html += '<tr>';
    for (const col of cols) {
      const value = profile[col.key];
      const align = col.align || 'right';
      const cell = formatCell(value, col.format);
      html += `<td class="col-${align} ${cell.className}">${cell.text}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody>';

  html += '</table>';
  container.innerHTML = html;

  // Attach click handlers to header cells
  const headers = container.querySelectorAll('th[data-key]');
  headers.forEach(th => {
    th.addEventListener('click', () => {
      const key = th.dataset.key;
      handleSort(key);
    });
  });
}

// Format cell value based on format type
function formatCell(value, format) {
  if (value == null) {
    return { text: '—', className: 'cell-muted' };
  }

  switch (format) {
    case 'risk':
      return formatRisk(value);
    case 'ndl':
      return formatNDL(value);
    case 'ceil':
      return formatCeiling(value);
    case 'delta_risk':
      return formatDelta(value, 'risk', 3);
    case 'delta_ndl':
      return formatDelta(value, 'ndl', 1);
    case 'delta_ceil':
      return formatDelta(value, 'ceil', 1);
    case 'depth':
      return { text: `${value.toFixed(1)}m`, className: '' };
    case 'time':
      return { text: `${value.toFixed(0)}min`, className: '' };
    case 'text':
      return { text: value, className: '' };
    default:
      return { text: String(value), className: '' };
  }
}

function formatRisk(value) {
  const formatted = value.toFixed(3);
  let className = '';
  if (value < 0.5) {
    className = 'cell-green';
  } else if (value < 1.0) {
    className = 'cell-yellow';
  } else {
    className = 'cell-red';
  }
  return { text: formatted, className };
}

function formatNDL(value) {
  if (value >= 100) {
    return { text: '100+', className: 'cell-muted' };
  }
  return { text: value.toFixed(1), className: '' };
}

function formatCeiling(value) {
  if (value === 0 || value === 0.0) {
    return { text: '—', className: 'cell-muted' };
  }
  return { text: value.toFixed(1), className: '' };
}

function formatDelta(value, type, decimals) {
  const sign = value >= 0 ? '+' : '';
  const formatted = `${sign}${value.toFixed(decimals)}`;

  let className = '';
  if (type === 'risk' || type === 'ceil') {
    // Positive delta = slab more conservative (green)
    className = value > 0 ? 'cell-green' : (value < 0 ? 'cell-red' : '');
  } else if (type === 'ndl') {
    // Negative delta = slab more conservative (green)
    className = value < 0 ? 'cell-green' : (value > 0 ? 'cell-red' : '');
  }

  return { text: formatted, className };
}

// Render pagination controls
function renderPagination(totalProfiles, currentPage, totalPages) {
  const container = document.getElementById('profile-pagination');
  if (!container) return;

  if (totalPages <= 1) {
    container.innerHTML = `<div class="pagination-info">Page 1 of 1 (${totalProfiles} profiles)</div>`;
    return;
  }

  let html = '<div class="pagination">';

  // Info text
  html += `<div class="pagination-info">Page ${currentPage + 1} of ${totalPages} (${totalProfiles} profiles)</div>`;

  // Buttons
  html += '<div class="pagination-buttons">';

  // Prev button
  const prevDisabled = currentPage === 0;
  html += `<button class="page-btn" data-page="${currentPage - 1}" ${prevDisabled ? 'disabled' : ''}>Prev</button>`;

  // Page buttons
  const pageButtons = buildPageButtons(currentPage, totalPages);
  for (const btn of pageButtons) {
    if (btn.type === 'ellipsis') {
      html += '<span class="page-ellipsis">...</span>';
    } else {
      const active = btn.page === currentPage ? 'active' : '';
      html += `<button class="page-btn ${active}" data-page="${btn.page}">${btn.page + 1}</button>`;
    }
  }

  // Next button
  const nextDisabled = currentPage === totalPages - 1;
  html += `<button class="page-btn" data-page="${currentPage + 1}" ${nextDisabled ? 'disabled' : ''}>Next</button>`;

  html += '</div></div>';
  container.innerHTML = html;

  // Attach click handlers
  const buttons = container.querySelectorAll('.page-btn:not([disabled])');
  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      const page = parseInt(btn.dataset.page, 10);
      handlePageChange(page);
    });
  });
}

// Build page button array with ellipsis
function buildPageButtons(currentPage, totalPages) {
  const buttons = [];
  const maxVisible = 7;

  if (totalPages <= maxVisible) {
    // Show all pages
    for (let i = 0; i < totalPages; i++) {
      buttons.push({ type: 'page', page: i });
    }
    return buttons;
  }

  // Always show first page
  buttons.push({ type: 'page', page: 0 });

  // Calculate range around current page
  let start = Math.max(1, currentPage - 2);
  let end = Math.min(totalPages - 2, currentPage + 2);

  // Adjust range if near edges
  if (currentPage <= 3) {
    end = maxVisible - 2;
  }
  if (currentPage >= totalPages - 4) {
    start = totalPages - maxVisible + 1;
  }

  // Add ellipsis if needed
  if (start > 1) {
    buttons.push({ type: 'ellipsis' });
  }

  // Add middle pages
  for (let i = start; i <= end; i++) {
    buttons.push({ type: 'page', page: i });
  }

  // Add ellipsis if needed
  if (end < totalPages - 2) {
    buttons.push({ type: 'ellipsis' });
  }

  // Always show last page
  buttons.push({ type: 'page', page: totalPages - 1 });

  return buttons;
}

// Event handlers
function handleSort(key) {
  const newDir = (_tableState.sortKey === key && _tableState.sortDir === 'asc') ? 'desc' : 'asc';
  _tableState.sortKey = key;
  _tableState.sortDir = newDir;
  _tableState.page = 0; // Reset to first page on sort

  if (_onStateChange) {
    _onStateChange({ ..._tableState });
  }
}

function handlePageChange(page) {
  _tableState.page = page;

  if (_onStateChange) {
    _onStateChange({ ..._tableState });
  }
}
