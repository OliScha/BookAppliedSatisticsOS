Einleitung
============================

## Aufgabenstellung

In diesem Projekt sollen Modelle für ein imaginäres Immobilienunternehmen entwickelt werden, mit deren Hilfe folgende Vorhersagen getroffen werden können:

1. Der durchschnittliche Preis von Immobilien in einem beliebigen Distrikt

2. Ob die durchschnittlichen Immobilienpreise in einem Distrikt über oder unter einem Wert von 150.000$ liegen

Um die erste Vorhersage zu treffen werden Regressionsmodelle verwendet, für die zweit Vorhersage Klassifikationsmodelle.

Die Kommentare zu Vorgehensweise und Überlegungen sowie den Code mit den entpsrechenden Ergebnissen befinden sich in den nächsten Kapiteln.

```{note}
Zur Modellierung werden Modelle von scikit learn und statsmodels verwendet.
```

## Zielsetzung

1. Regression:

Da es sich bei Haus-/Wohnungspreisen in diesem Fall um niedrige bis höhere 6-stellige Summen handelt (ca. 100.000 - 500.000$ in diesem Fall), wäre ein Modell mit einem 4-stelligen bis niedrig 5-stelligen RMSE wünschenswert.

2. Klassifikation:

Da für diesen Anwendungsfall sowohl die Anzahl an als "above" oder "below" identifizierten Distrikten relevant ist als auch die Korrekheit dieser Angabe, beides aber nicht über Leben und Tod entscheidet, soll der F1-Score als Kriterium genutzt werden. Ein F1-Score von über 80% wäre wünschenswert.
