# Fazit

## Klassifikation

asdasf

## Klassifikation

Das Classification Modell von statsmodels erzielt in diesem Fall leicht bessere Ergebnisse als das von scikit learn und würde daher für das Deployment in eine produktive Umgebung ausgewählt werden.

Es fällt auf, dass die Abweichungen der Modellergebnisse zwischen Trainings- und Testdaten nicht so groß sind wie bei den Regressionsmodellen. Das liegt möglicherweise daran, dass hier nicht so viele Outlier entfernt wurden wie bspw. durch Cook's Distance oder Regression Diagnostics.

Auch das Ergebnis generell ist um einiges zufriedenstellender als das der Regressionsmodelle, was wohl aber an der Art der Vorhersage liegt (Es ist natürlich einfacher einen binären Wert wie Ja/Nein oder in diesem Fall über/unter vorherzusagen als einen exakten Preis).
