// content.js
(function () {
  // Helfer: Domain bestimmen
  function getDomain(url) {
    try {
      const u = new URL(url)
      return u.hostname
    } catch (e) {
      return location.hostname || 'unknown'
    }
  }

  // URL-View einmal beim Laden melden
  chrome.runtime.sendMessage({
    type: 'CT_URL_VIEW',
    url: location.href,
    domain: getDomain(location.href),
  })

  // Klicks zählen
  document.addEventListener(
    'click',
    () => {
      const domain = getDomain(location.href)
      chrome.runtime.sendMessage({ type: 'CT_CLICK', domain })
    },
    { capture: true }
  )
})()
