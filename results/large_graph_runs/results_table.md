# Dissertation Results — All Graphs & Methods

Generated: 2026-06-22

## Network Statistics

| Graph | Nodes | Edges | Components |
|---|---|---|---|
| Flickr | 1,624,991 | 15,473,043 | 1 |
| Youtube | 1,134,890 | 2,987,624 | 1 |
| Epinions | 75,877 | 405,739 | 1 |
| Facebook | 63,392 | 816,831 | 1 |
| Gnutella31 | 62,561 | 147,878 | 1 |
| Enron | 33,696 | 180,811 | 1 |
| Digg | 29,652 | 84,781 | 1 |
| HI-II-14 | 4,165 | 13,087 | 1 |
| Crime | 829 | 1,473 | 1 |

## Results — ANC (lower = better)

| Graph | Nodes | Budget | Degree | HDA | CI | FINDER (Original) | MUSE-CN (MTSSL MEGA) |
|---|---|---|---|---|---|---|---|
| Flickr | 1,624,991 | 162,499 | 0.4701 | 0.3795 | 0.4106 | 0.3735 | **0.3263** |
| Youtube | 1,134,890 | 113,489 | 0.1488 | **0.1191** | 0.1251 | 0.1486 | 0.1395 |
| Epinions | 75,877 | 7,587 | 0.3509 | 0.3218 | 0.3324 | 0.3362 | **0.3013** |
| Facebook | 63,392 | 6,339 | 0.8802 | 0.8768 | 0.8784 | 0.8801 | **0.8142** |

## Results — end\_PC (lower = better)

| Graph | Nodes | Budget | Degree | HDA | CI | FINDER (Original) | MUSE-CN (MTSSL MEGA) |
|---|---|---|---|---|---|---|---|
| Flickr | 1,624,991 | 162,499 | 0.0823 | 0.0382 | **0.0189** | 0.0528 | 0.0393 |
| Youtube | 1,134,890 | 113,489 | 0.0000 | **0.0000** | **0.0000** | 0.0000 | 0.0000 |
| Epinions | 75,877 | 7,587 | 0.0363 | 0.0049 | **0.0001** | 0.0220 | 0.0156 |
| Facebook | 63,392 | 6,339 | 0.7630 | **0.7556** | 0.7588 | 0.7678 | 0.6606 |

## Notes

- **ANC** = Average Node Connectivity (area under the connectivity curve during node removal, normalized). Lower = more connectivity destroyed.
- **end\_PC** = end Pairwise Connectivity (fraction of node pairs still reachable after removal). Lower = better attack.
- **Budget** = 10% of node count (standard CNDP setting).
- **Bold** values indicate the best (lowest) result per graph per metric.
- **FINDER** = ORIGINAL\_BA\_FINDER (learning baseline).
- **MUSE-CN** = MTSSL\_MEGA\_CrossAttention\_1l\_freeze\_BA\_FINDER\_noProc (proposed model).
- All graphs are single connected components.
