// options.js
const consentEl = document.getElementById('consent')
const txt = document.getElementById('data')
const refreshBtn = document.getElementById('refresh')
const exportBtn = document.getElementById('export')
const resetBtn = document.getElementById('reset')

function load() {
  chrome.storage.local.get(null, (res) => {
    consentEl.checked = !!res.ct_consent
    const dump = {
      consent: !!res.ct_consent,
      clicksPerDomain: res.ct_data || {},
      pageViews: res.ct_pages || {},
    }
    txt.value = JSON.stringify(dump, null, 2)
  })
}

consentEl.addEventListener('change', () => {
  chrome.storage.local.set({ ct_consent: consentEl.checked }, load)
})

refreshBtn.addEventListener('click', load)

exportBtn.addEventListener('click', () => {
  chrome.storage.local.get(null, (res) => {
    const dump = {
      exportCreatedAt: new Date().toISOString(),
      consent: !!res.ct_consent,
      clicksPerDomain: res.ct_data || {},
      pageViews: res.ct_pages || {},
    }
    const blob = new Blob([JSON.stringify(dump, null, 2)], {
      type: 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'click-tracker-export.json'
    a.click()
    URL.revokeObjectURL(url)
  })
})

resetBtn.addEventListener('click', () => {
  if (confirm('Wirklich alle lokal gespeicherten Daten löschen?')) {
    chrome.storage.local.set({ ct_data: {}, ct_pages: {} }, load)
  }
})

load()
