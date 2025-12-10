# Resource Availability Models - Visual Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RESOURCE AVAILABILITY SYSTEM                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌────────────────────┐         ┌────────────────────────┐
        │   BASIC MODEL      │         │   ADVANCED MODEL       │
        │  (Interval-Based)  │         │  (Pattern Mining)      │
        └────────────────────┘         └────────────────────────┘
                    │                               │
                    │                               ├─────────────────┬───────────────┐
                    │                               │                 │               │
                    ▼                               ▼                 ▼               ▼
        ┌────────────────────┐         ┌────────────────┐  ┌────────────────┐  ┌──────────────┐
        │  • 2-week cycles   │         │ Pattern Mining │  │   Clustering   │  │  Lifecycle   │
        │  • Fixed hours     │         │  • Work hours  │  │  • K-means     │  │  Tracking    │
        │  • Holiday check   │         │  • Peak hours  │  │  • 5 clusters  │  │  • Busy check│
        │  • Next available  │         │  • Intensity   │  │  • Profiles    │  │  • Current   │
        └────────────────────┘         └────────────────┘  └────────────────┘  │    activity  │
                    │                               │                 │         └──────────────┘
                    │                               ▼                 │               │
                    │                   ┌────────────────────┐        │               │
                    │                   │  Probabilistic     │        │               │
                    │                   │  Predictions       │        │               │
                    │                   │  • Hour prob       │◄───────────────────────┘
                    │                   │  • Day prob        │  (0% if busy)
                    │                   │  • Combined        │
                    │                   └────────────────────┘
                    │                               │
                    └───────────────┬───────────────┴─────────────────┘
                                    ▼
                        ┌────────────────────────┐
                        │  SIMULATION ENGINE     │
                        │  • Check availability  │
                        │  • Find resources      │
                        │  • Schedule activities │
                        └────────────────────────┘
```

## Data Flow

```
BPIC 2017 Dataset (XES)
         │
         ▼
┌─────────────────┐
│  Load & Parse   │
│  (PM4Py)        │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Event Log DataFrame            │
│  • org:resource                 │
│  • time:timestamp               │
│  • concept:name                 │
│  • case:concept:name            │
│  • lifecycle:transition (NEW)   │
└────────┬────────────────────────┘
         │
         ├──────────────────────────┐
         │                          │
         ▼                          ▼
┌─────────────────┐    ┌──────────────────────┐
│  Basic Model    │    │  Advanced Model      │
│  Configuration  │    │  Pattern Mining      │
└────────┬────────┘    └──────────┬───────────┘
         │                        │
         │                        ├─────────────┬──────────────┐
         │                        │             │              │
         ▼                        ▼             ▼              ▼
┌─────────────────┐    ┌──────────────┐  ┌────────────┐  ┌──────────────┐
│ Availability    │    │  Resource    │  │  Resource  │  │ Busy Period  │
│ Checker         │    │  Patterns    │  │  Clusters  │  │ Extractor    │
└─────────────────┘    └──────────────┘  └────────────┘  └──────┬───────┘
                              │                                  │
                              ▼                                  ▼
                    ┌──────────────────┐              ┌──────────────────┐
                    │  Probability     │◄─────────────│ Lifecycle Checker│
                    │  Calculator      │  (blocks if  │ • is_busy_at()   │
                    └──────────────────┘   busy)      └──────────────────┘
```

## Pattern Mining Process

```
For Each Resource:
    
    1. EXTRACT ACTIVITIES & LIFECYCLE EVENTS (NEW)
       ┌──────────────────────────┐
       │ Filter by org:resource   │
       │ Get all timestamps       │
       │ Extract lifecycle states │
       │ Match start→complete     │
       └──────────┬───────────────┘
                  │
                  ▼
    2. TEMPORAL ANALYSIS
       ┌──────────────────────────┐
       │ • Hour distribution      │
       │ • Day-of-week distribution│
       │ • Date range            │
       └──────────┬───────────────┘
                  │
                  ▼
    3. PATTERN EXTRACTION
       ┌──────────────────────────┐
       │ • Working hours (min/max)│
       │ • Peak hours (top 25%)   │
       │ • Activity intensity     │
       │ • Working days (unique)  │
       │ • Busy periods (NEW)     │
       └──────────┬───────────────┘
                  │
                  ▼
    4. PROBABILITY CALCULATION
       ┌──────────────────────────┐
       │ • Normalize distributions│
       │ • Calculate probabilities│
       │ • Store in pattern dict  │
       └──────────────────────────┘
```

## Clustering Workflow

```
All Resources with sufficient data (>10 activities)
                  │
                  ▼
    ┌────────────────────────────┐
    │  FEATURE EXTRACTION        │
    │  Per Resource:             │
    │  • working_start           │
    │  • working_end             │
    │  • num_working_days        │
    │  • log(activity_intensity) │
    │  • hour_spread             │
    │  • day_spread              │
    └──────────┬─────────────────┘
               │
               ▼
    ┌────────────────────────────┐
    │  K-MEANS CLUSTERING        │
    │  n_clusters = 5            │
    │  random_state = 42         │
    └──────────┬─────────────────┘
               │
               ├────────┬────────┬────────┬────────┐
               ▼        ▼        ▼        ▼        ▼
          Cluster 0 Cluster 1 Cluster 2 Cluster 3 Cluster 4
          29 res.   72 res.   8 res.    3 res.    34 res.
          Early     Flexible  Office    High-Int  Extended
          Starters  Hours     Workers   Workers   Hours
```

## Availability Check Logic

```
is_available(resource_id, current_time)
    │
    ├─> Is resource in system?
    │   NO → Return False
    │   YES → Continue
    │
    ├─> [LIFECYCLE CHECK - NEW] Is resource busy with activity?
    │   YES → Return False (probability = 0%)
    │   NO → Continue
    │
    ├─> Is it a public holiday?
    │   YES → Return False
    │   NO → Continue
    │
    ├─> [ADVANCED ONLY] Is within activity date range?
    │   NO → Return False
    │   YES → Continue
    │
    ├─> Is current day a working day?
    │   NO → Return False
    │   YES → Continue
    │
    ├─> Is current hour within working hours?
    │   NO → Return False
    │   YES → Continue
    │
    └─> [PROBABILISTIC MODE] Random < probability?
        NO → Return False
        YES → Return True
        
    [DEFAULT MODE] → Return True
```

## BPIC 2017 Cluster Visualization

```
Cluster Distribution:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cluster 0 (29): ████████████                    Early Starters
                 5:30-17:30, 95.39 act/day

Cluster 1 (72): ████████████████████████████    Flexible Hours
                 6:00-20:45, 92.06 act/day       (LARGEST)

Cluster 2 (8):  ████                            Office Workers
                 8:00-16:00, 55.04 act/day       (Mon-Fri only)

Cluster 3 (3):  ██                              High-Intensity
                 1:00-23:20, 172.82 act/day      (Near 24/7)

Cluster 4 (34): ██████████████                  Extended Hours
                 7:00-20:00, 74.21 act/day
```

## Working Hours Distribution

```
Hour of Day Activity (All Resources):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

00:00  ▏ 569
01:00  ▏ 339
02:00  ▏ 314
03:00  ▏ 335
04:00  ▏ 978
05:00  █ 12,372
06:00  ████ 52,235
07:00  ████████ 94,066
08:00  ███████████ 123,168  ◄── PEAK START
09:00  ████████████ 133,231 ◄── HIGHEST
10:00  ██████████ 116,259
11:00  ████████ 100,854
12:00  █████████ 108,987
13:00  ██████████ 114,883
14:00  █████████ 103,022   ◄── PEAK END
15:00  █████ 65,402
16:00  ███ 39,364
17:00  ███ 40,063
18:00  ████ 47,605
19:00  ██ 27,300
20:00  ▏ 8,617
21:00  ▏ 5,136
22:00  ▏ 4,591
23:00  ▏ 2,577
```

## Day of Week Distribution

```
Activity by Weekday:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Monday    ████████████████████ 248,785 (20.7%)
Tuesday   ███████████████████ 226,559 (18.8%)
Wednesday ██████████████████ 223,321 (18.6%)
Thursday  ████████████████ 200,183 (16.6%)
Friday    ████████████████ 202,811 (16.9%)
Saturday  ██████ 80,019 (6.7%)
Sunday    █ 20,589 (1.7%)
```

## Probability Calculation Example

```
Resource: User_3 at Wednesday 9:00 AM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Get Hour Probability
        hour_prob[9] = 0.087 (8.7% of activities at 9 AM)

Step 2: Get Day-of-Week Probability
        dow_prob[2] = 0.168 (16.8% of activities on Wednesday)

Step 3: Calculate Combined Probability
        combined = sqrt(0.087 × 0.168) = 0.121 (12.1%)

Step 4: Check Peak Hour
        Hour 9 IS in peak_hours [8, 9, 10, 13]
        Boost: 0.121 × 1.3 = 0.157

Step 5: Cap at 1.0
        final_prob = min(1.0, 0.157) = 0.157

Result: 15.7% probability ✓
```

## API Usage Flow

```
┌─────────────────────────────────────────────────────────┐
│  SIMULATION ENGINE                                      │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
        Need to assign activity to resource
                        │
                        ▼
        ┌───────────────────────────────┐
        │  Get candidate resources      │
        │  for this activity type       │
        └───────────┬───────────────────┘
                    │
                    ▼
        ┌───────────────────────────────┐
        │  For each candidate:          │
        │  is_available(res, time)      │
        └───────────┬───────────────────┘
                    │
        ┌───────────┴──────────┐
        │                      │
        ▼                      ▼
    Available              Not Available
        │                      │
        ▼                      ▼
    Select best       get_next_available_time()
    resource                   │
        │                      │
        └──────────┬───────────┘
                   ▼
        ┌──────────────────────┐
        │  Assign & Schedule   │
        └──────────────────────┘
```

## Key Metrics Summary

```
┌────────────────────────────────────────────────────────┐
│  BPIC 2017 RESOURCE AVAILABILITY METRICS               │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Total Resources:        149                           │
│  Total Events:           1,202,267                     │
│  Date Range:             396 days                      │
│                                                        │
│  Lifecycle Tracking (NEW):                             │
│    States Found:         7 lifecycle states            │
│    Busy Periods:         36,776 extracted              │
│    Resources Tracked:    143 (96%)                     │
│    Avg Busy Duration:    59.93 hours                   │
│    Avg Periods/Resource: 257.2                         │
│                                                        │
│  Clusters Identified:    5                             │
│  Patterns Mined:         149 (100%)                    │
│                                                        │
│  Average Availability:                                 │
│    Wednesday 10 AM:      94 resources (63%)            │
│    Saturday 10 AM:       Lower availability            │
│                                                        │
│  Top Resource:           User_1                        │
│    Activities:           148,404                       │
│    Pattern:              24/7 (Automated)              │
│    Cluster:              3 (High-Intensity)            │
│    Busy Periods:         11,950 tracked                │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## Implementation Completeness

```
Feature Checklist:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BASIC MODEL:
  ✅ Interval-based availability
  ✅ Configurable working hours
  ✅ Configurable cycle days
  ✅ Holiday integration
  ✅ Next available time
  ✅ Timezone support

ADVANCED MODEL:
  ✅ Pattern mining from historical data
  ✅ Individual resource profiles
  ✅ Working hour detection
  ✅ Peak hour identification
  ✅ Activity intensity calculation
  ✅ K-means clustering (5 clusters)
  ✅ Cluster profile generation
  ✅ Probabilistic predictions
  ✅ Workload statistics
  ✅ Available resources query
  ✅ Resource info API
  ✅ Timezone handling
  ✅ Lifecycle-aware availability (NEW)
  ✅ Busy period extraction (36,776 periods)
  ✅ Real-time busy checking
  ✅ Current activity detection
  ✅ Workload calculation (overlapping activities)
  ✅ Automatic double-booking prevention

TESTING:
  ✅ Comprehensive test suite
  ✅ Basic model tests (13 cases)
  ✅ Advanced model tests
  ✅ Lifecycle tracking tests (NEW)
  ✅ Busy period validation (NEW)
  ✅ Integration examples
  ✅ All tests passing

DOCUMENTATION:
  ✅ API documentation
  ✅ Usage examples
  ✅ Quick start guide
  ✅ Implementation summary
  ✅ Visual diagrams
```

---

**Status**: ✅ Complete and Production-Ready
