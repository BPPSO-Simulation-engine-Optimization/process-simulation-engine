// popup.js
const tbody = document.getElementById('table-body')
const consentEl = document.getElementById('consent')
const resetBtn = document.getElementById('reset')
const optionsLink = document.getElementById('options')

function render(data) {
  tbody.innerHTML = ''
  const entries = Object.entries(data || {}).sort((a, b) => b[1] - a[1])
  for (const [domain, count] of entries) {
    const tr = document.createElement('tr')
    const tdD = document.createElement('td')
    tdD.textContent = domain
    const tdC = document.createElement('td')
    tdC.textContent = count
    tr.appendChild(tdD)
    tr.appendChild(tdC)
    tbody.appendChild(tr)
  }
}

function load() {
  chrome.storage.local.get(['ct_consent', 'ct_data'], (res) => {
    consentEl.checked = !!res.ct_consent
    render(res.ct_data || {})
  })
}

consentEl.addEventListener('change', () => {
  chrome.storage.local.set({ ct_consent: consentEl.checked })
  load()
})

resetBtn.addEventListener('click', () => {
  chrome.storage.local.set({ ct_data: {}, ct_pages: {} }, load)
})

optionsLink.addEventListener('click', (e) => {
  e.preventDefault()
  chrome.runtime.openOptionsPage()
})

load()
