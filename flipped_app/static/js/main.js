// ── Dark / Light Mode Toggle ────────────────────────────
(function () {
  const html     = document.documentElement;
  const btn      = document.getElementById('darkModeToggle');
  const icon     = document.getElementById('darkModeIcon');
  const label    = document.getElementById('darkModeLabel');

  function applyTheme(theme) {
    html.setAttribute('data-theme', theme);
    localStorage.setItem('fliplearnTheme', theme);
    if (theme === 'dark') {
      if (icon)  { icon.className  = 'bi bi-sun-fill'; }
      if (label) { label.textContent = 'Light'; }
    } else {
      if (icon)  { icon.className  = 'bi bi-moon-stars-fill'; }
      if (label) { label.textContent = 'Dark'; }
    }
  }

  // Initialise from saved pref (already set by inline script, just sync UI)
  const saved = localStorage.getItem('fliplearnTheme') || 'light';
  applyTheme(saved);

  if (btn) {
    btn.addEventListener('click', function () {
      const current = html.getAttribute('data-theme') || 'light';
      applyTheme(current === 'dark' ? 'light' : 'dark');
    });
  }
})();

// ── Basic UI helpers ────────────────────────────────────
(function () {
  const alerts = document.querySelectorAll('.alert');
  alerts.forEach(a => {
    setTimeout(() => a.classList.add('fade'), 8000);
  });
})();

// ── Auto-Logout after 5 minutes of inactivity ───────────
(function () {
  const cfg = window.FLIPLEARN;
  if (!cfg) return;  // Not logged in — skip

  const logoutForm   = document.getElementById(cfg.logoutFormId);
  const toastEl      = document.getElementById(cfg.toastId);
  const countdownEl  = document.getElementById(cfg.countdownId);
  const stayBtn      = document.getElementById(cfg.stayBtnId);
  if (!logoutForm || !toastEl) return;

  const bsToast = new bootstrap.Toast(toastEl, { autohide: false });
  let idleTimer    = null;   // fires when warning should show
  let countdownInterval = null;
  let secondsLeft  = 0;

  // Show the warning toast and start 60-second countdown
  function showWarning() {
    secondsLeft = Math.floor(cfg.WARNING_BEFORE / 1000);
    countdownEl.textContent = secondsLeft;
    bsToast.show();

    countdownInterval = setInterval(function () {
      secondsLeft -= 1;
      countdownEl.textContent = secondsLeft;
      if (secondsLeft <= 0) {
        clearInterval(countdownInterval);
        logoutForm.submit();  // Auto-logout
      }
    }, 1000);
  }

  // Reset everything — called on any user activity
  function resetTimer() {
    clearTimeout(idleTimer);
    clearInterval(countdownInterval);
    bsToast.hide();

    // Schedule warning at (TIMEOUT - WARNING_BEFORE)
    const warnAfter = cfg.INACTIVITY_TIMEOUT - cfg.WARNING_BEFORE;
    idleTimer = setTimeout(showWarning, warnAfter);
  }

  // Stay Logged In button
  stayBtn.addEventListener('click', function () {
    resetTimer();
    // Also ping the server so SESSION_SAVE_EVERY_REQUEST resets the cookie
    fetch(window.location.href, { method: 'HEAD', credentials: 'same-origin' });
  });

  // Track any user activity
  ['mousemove', 'mousedown', 'keydown', 'scroll', 'touchstart', 'click']
    .forEach(evt => document.addEventListener(evt, resetTimer, { passive: true }));

  // Kick off the first timer
  resetTimer();
})();
