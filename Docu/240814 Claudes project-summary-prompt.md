# LRH Textanalyse-Projekt: Erweitertes Modell-Ensemble

## Projektziel
Entwicklung eines robusten Klassifizierungssystems zur Unterscheidung authentischer L. Ron Hubbard (LRH) Texte von nicht-authentischen Texten, basierend auf einem erweiterten Ensemble von Sprachmodellen.

## Aktuelle Erkenntnisse
- Analyse von 26.000 Seiten zeigt abnehmenden Trend in der Authentizität über die Jahre.
- Identifizierung von 3.800 Seiten mit >80% und 7.400 Seiten mit >50% Übereinstimmung als potenzielle hochwertige Trainingsdaten.
- Mögliche Verwendung von Vortragsaufzeichnungen (1950-1972) als zusätzliche authentische Quelle.

## Neue Modelle zur Integration
1. GPT-3 oder GPT-J-6B: Für subtile Stilnuancen und autoregressive Analyse.
2. T5-large oder T5-3B: Vielseitig für spezifische NLP-Aufgaben.
3. LongFormer: Spezialisiert auf lange Texte für umfangreiche Dokumente.
4. XLNet-large: Permutationsbasierte Methode für verbesserte Kontextmodellierung.
5. ELECTRA-large: Effizient mit diskriminativem Vortraining.

## Nächste Schritte
1. Integration der neuen Modelle in das bestehende Ensemble.
2. Entwicklung einer Gewichtungsmethode für Modellvorhersagen.
3. Durchführung einer Fehleranalyse zur Identifikation von Modellstärken und -schwächen.
4. Experimentieren mit verschiedenen Trainingsdatensätzen (hochbewertete Seiten, Vortragsaufzeichnungen).
5. Optimierung des Gesamtensembles für maximale Genauigkeit und Zuverlässigkeit.

## Besondere Überlegungen
- Ressourcen (Rechenzeit) sind nicht limitierend aufgrund des verfügbaren Power-PCs.
- Fokus liegt auf der Schaffung einer ausgewogenen Mischung verschiedener Ansätze für robuste Ergebnisse.
