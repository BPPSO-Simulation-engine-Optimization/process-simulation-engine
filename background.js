// background.js (Service Worker, MV3)
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    chrome.storage.local.set({ ct_consent: false, ct_data: {} })
    // Optional: beim ersten Mal Options-Seite öffnen
    chrome.runtime.openOptionsPage()
  }
})

// Nachrichten vom Content Script empfangen und speichern
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'CT_CLICK') {
    // Nur speichern, wenn Einwilligung erteilt wurde
    chrome.storage.local.get(['ct_consent', 'ct_data'], (res) => {
      const consent = !!res.ct_consent
      if (!consent) return // ohne Consent keine Speicherung

      const data = res.ct_data || {}
      const key = msg.domain || 'unknown'
      data[key] = (data[key] || 0) + 1
      chrome.storage.local.set({ ct_data: data })
    })
  }
  if (msg.type === 'CT_URL_VIEW') {
    chrome.storage.local.get(['ct_consent', 'ct_pages'], (res) => {
      const consent = !!res.ct_consent
      if (!consent) return

      const pages = res.ct_pages || {}
      const key = msg.url || 'about:blank'
      pages[key] = (pages[key] || 0) + 1
      chrome.storage.local.set({ ct_pages: pages })
    })
  }
  // keep message channel open only if needed
  // sendResponse optional
  return false
})