# Simulation Benchmark Analysis - BPIC 2017

**Datum:** 29. Dezember 2025  
**Original Log:** BPI Challenge 2017.xes  
**Simulated Log:** simulated_log.xes  

---

## Gesamtbewertung: C+ (75/100)

### Stärken ✅
- **Hervorragende Prozessvariabilität:** 12,698 eindeutige Trace-Varianten (79.7% des Originals)
- **Korrekte Fallanzahl:** 31,500 Fälle (99.97% Übereinstimmung)
- **Variable Ankunftsraten:** Realistische tägliche/stündliche Schwankungen

### Kritische Schwächen ❌
- **77% zu schnelle Durchlaufzeiten:** 104h statt 458h
- **59% weniger Events pro Fall:** 15.5 statt 38.16 
- **85% fehlende Prozesspfade:** Nur 26 statt 178 DFG-Kanten
- **Falsche End-Aktivitäten:** Kritische Aktivitäten fehlen komplett

---

## 1. Basis-Statistiken

| Metrik | Original | Simuliert | Differenz |
|--------|----------|-----------|-----------|
| **Gesamte Events** | 1,202,267 | 486,883 | -59.5% |
| **Fälle** | 31,509 | 31,500 | -0.03% ✅ |
| **Aktivitäten** | 26 | 16 | -38.5% ⚠️ |
| **Ressourcen** | 149 | 105 | -29.5% |

### Fehlende Aktivitäten (10):
- `W_Shortened completion` (24,535 Events im Original)
- `A_Cancelled` (10,329 Events)
- `W_Validate application` (10,155 Events)
- `A_Approved` (7,367 Events)
- `A_Registered` (5,265 Events)
- `O_Cancelled` (3,655 Events)
- `O_Declined` (3,497 Events)
- `A_Finalized` (2,246 Events)
- `W_Assess potential fraud` (1,380 Events)
- `O_Sent (mail and online)` (1,376 Events)

**→ Impact:** 38% der Prozessaktivitäten fehlen komplett, was zu deutlich simplifizierten Prozessabläufen führt.

---

## 2. Events pro Fall

| Statistik | Original | Simuliert | Differenz |
|-----------|----------|-----------|-----------|
| **Mittelwert** | 38.16 | 15.46 | -59.5% ⚠️ |
| **Median** | 35.0 | 13.0 | -62.9% |
| **Std. Abw.** | 19.76 | 13.44 | -32.0% |
| **Min** | 3 | 2 | -33.3% |
| **Max** | 180 | 132 | -26.7% |

**→ Kritisch:** Die Simulation erzeugt nur 40% der erwarteten Events. Dies deutet auf:
- Fehlende Rework-Schleifen
- Vereinfachte Prozesspfade
- Zu wenig Wiederholungen von Aktivitäten

---

## 3. Durchlaufzeiten

| Statistik | Original | Simuliert | Differenz |
|-----------|----------|-----------|-----------|
| **Median** | 458.43h (19 Tage) | 104.22h (4.3 Tage) | **-77.3%** ⚠️⚠️⚠️ |
| **Mittelwert** | 685.03h (28.5 Tage) | 152.07h (6.3 Tage) | -77.8% |
| **Std. Abw.** | 576.41h | 129.25h | -77.6% |

**→ Kritischster Fehler:** Die Simulation ist **4x zu schnell**. Ursachen:
- Keine oder unzureichende Wartezeiten (Queue Times)
- Fehlende Bearbeitungszeiten zwischen Aktivitäten
- Zu optimistische Ressourcenverfügbarkeit

---

## 4. Trace-Varianten

| Metrik | Original | Simuliert | Coverage |
|--------|----------|-----------|----------|
| **Eindeutige Varianten** | 15,930 | 12,698 | 79.7% ✅ |
| **Top-1 Variante** | 1.71% | 3.53% | - |
| **Top-5 Varianten** | 5.42% | 10.10% | - |
| **Top-10 Varianten** | 7.58% | 13.39% | - |

**→ Positiv:** Die Simulation erzeugt fast 80% der ursprünglichen Varianz - **hervorragend** für die Prozessvariabilität!

---

## 5. Directly-Follows Graph (DFG)

| Metrik | Original | Simuliert | Coverage |
|--------|----------|-----------|----------|
| **DFG-Kanten** | 178 | 26 | **14.6%** ⚠️⚠️⚠️ |

### Fehlende kritische Pfade:
Nur 26 von 178 Prozesspfaden sind implementiert. **85% der Prozesspfade fehlen komplett!**

**→ Kritisch:** Die Simulation deckt nur einen Bruchteil der möglichen Prozessflüsse ab. Dies führt zu:
- Deutlich vereinfachtem Prozessmodell
- Fehlenden alternativen Pfaden
- Unrealistischer Prozessstruktur

---

## 6. Aktivitäts-Zeitdauern

### Beispiele für Abweichungen:

| Aktivität | Original (Median) | Simuliert (Median) | Status |
|-----------|-------------------|---------------------|---------|
| `O_Created` | 0.00h | 9.84h | ⚠️ Zu langsam |
| `O_Sent` | 0.25h | 0.02h | ✅ Akzeptabel |
| `W_Complete application` | 46.67h | 63.09h | ⚠️ 35% zu langsam |
| `W_Call incomplete files` | 42.63h | 20.22h | ⚠️ 53% zu schnell |
| `A_Accepted` | 0.00h | 9.88h | ⚠️ Sollte instant sein |

**→ Problem:** Viele Aktivitäten haben unrealistische Dauern:
- Instant-Aktivitäten (O_Created, A_Accepted) dauern plötzlich ~10h
- Längere Aktivitäten haben inkonsistente Zeiten

---

## 7. End-Aktivitäten (KRITISCH)

| Aktivität | Original | Simuliert | Status |
|-----------|----------|-----------|---------|
| `W_Validate application` | **40.0%** | **0.0%** | ❌ Fehlt komplett! |
| `W_Call after offers` | 30.1% | 63.5% | ⚠️ Überrepräsentiert |
| `O_Cancelled` | **14.3%** | **0.0%** | ❌ Fehlt komplett! |
| `O_Accepted` | 6.3% | 10.2% | ⚠️ Zu häufig |
| `O_Declined` | 5.1% | 0.0% | ❌ Fehlt komplett! |

**→ Kritischster Fehler:** Die häufigsten End-Aktivitäten des Originals fehlen komplett:
- **W_Validate application** (40% → 0%) - Hauptabschluss fehlt!
- **O_Cancelled** (14% → 0%) - Abbruchpfad fehlt!
- **O_Declined** (5% → 0%) - Ablehnungspfad fehlt!

Stattdessen endet die Simulation in 63.5% mit `W_Call after offers`, was nur 30% im Original ausmacht.

---

## 8. Ressourcen-Verteilung

### Top-Ressourcen (Original):
1. User_10: 2.49%
2. User_4: 2.36%
3. User_8: 2.33%

### Top-Ressourcen (Simuliert):
1. **User_1: 17.86%** ⚠️⚠️⚠️ (68% aller Aktivitäten!)
2. User_3: 8.19%
3. User_4: 6.87%

**→ Kritisch:** User_1 ist massiv überlastet mit 68% aller Aktivitäten (sollte ~2% sein).

**Problem:** 
- Extrem ungleiche Verteilung
- User_1 führt 68% aller `W_Call after offers` aus
- Unrealistische Ressourcenzuteilung

---

## 9. Prioritisierte Verbesserungsvorschläge

### 🔴 Priorität 1: Durchlaufzeiten korrigieren
**Problem:** 77% zu schnell (104h statt 458h)  
**Lösung:**
- Queue Times zwischen Aktivitäten erhöhen
- Wartezeiten für Ressourcenverfügbarkeit einbauen
- Realistische Bearbeitungszeiten implementieren

### 🔴 Priorität 2: End-Aktivitäten korrigieren
**Problem:** Hauptabschlüsse fehlen komplett  
**Lösung:**
- `W_Validate application` als Hauptabschluss (40%) implementieren
- `O_Cancelled` Abbruchpfad (14%) hinzufügen
- `W_Call after offers` auf 30% reduzieren

### 🟠 Priorität 3: Events pro Fall erhöhen
**Problem:** 59% zu wenig (15.5 statt 38)  
**Lösung:**
- Rework-Schleifen einbauen
- Wiederholungen von `W_Call incomplete files` erhöhen
- Mehr alternative Pfade aktivieren

### 🟠 Priorität 4: Instant-Aktivitäten korrigieren
**Problem:** O_Created, A_Accepted dauern ~10h statt 0h  
**Lösung:**
- Diese Aktivitäten auf <1h setzen
- Automatische Aktivitäten ohne Wartezeit implementieren

### 🟡 Priorität 5: Ressourcen-Verteilung balancieren
**Problem:** User_1 überlastet (68% statt 2%)  
**Lösung:**
- Ressourcen-Pool für `W_Call after offers` erweitern
- Round-robin Zuteilung implementieren
- Realistische Kapazitätsgrenzen setzen

### 🟡 Priorität 6: Fehlende Aktivitäten implementieren
**Problem:** 10 Aktivitäten fehlen (38%)  
**Lösung:**
- `W_Shortened completion` (häufigste fehlende Aktivität)
- `A_Cancelled`, `A_Approved`, `A_Registered`
- Fraud-Detection Pfad mit `W_Assess potential fraud`

---

## 10. Zusammenfassung

### Was funktioniert gut ✅
1. **Prozessvariabilität:** 12,698 verschiedene Trace-Varianten zeigen exzellente Diversität
2. **Fallanzahl:** Nahezu perfekte Übereinstimmung (31,500 vs 31,509)
3. **Ankunftsraten:** Realistische tägliche und stündliche Schwankungen

### Was muss dringend verbessert werden ❌
1. **Durchlaufzeiten:** 4x zu schnell - kritischster Fehler
2. **End-Aktivitäten:** Hauptabschlüsse fehlen komplett
3. **Prozessabdeckung:** 85% der Prozesspfade fehlen
4. **Events pro Fall:** Nur 40% der erwarteten Prozessschritte

### Empfohlene nächste Schritte
1. Queue Times zwischen allen Aktivitäten einbauen → Durchlaufzeit erhöhen
2. End-Aktivitäts-Logik überarbeiten → W_Validate application als Hauptabschluss
3. Rework-Mechanismen implementieren → Mehr Events pro Fall
4. DFG-Analyse durchführen → Fehlende Prozesspfade identifizieren und implementieren
5. Ressourcen-Allokation überarbeiten → Gleichmäßigere Verteilung

---

**Fazit:** Die Simulation hat eine hervorragende Grundlage mit guter Prozessvariabilität, aber es gibt kritische Probleme bei den Zeitdauern, Prozessabschlüssen und der Prozessabdeckung. Mit den priorisierten Verbesserungen kann die Qualität deutlich gesteigert werden.
