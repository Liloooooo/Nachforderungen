# Nachforderungen

Dies ist ein Repository zum Training eines Regressionsmodells. 

Das Jupyter-Notebook in jupyter/ dient zu Testzwecken. Dort habe ich die einzelnen Arbeitsschritte dokumentiert, und ein Modell trainiert. Das beste Modell wurde im Pickle-Format in dem Ordner abgespeichert.  
Die Daten wurden nicht mit hochgeladen, und müssen gegebenenfalls ergänzt werden. 

Unter skript/ findet sich ein Minimalbeispiel einer Skriptstruktur. Die Modellarchitektur ist für den Zweck nicht variabel gestaltet. Das Skript dient lediglich zu Anschauungszwecken und könnte in einer ähnlichen Form zum Experiment Tracking genutzt werden. 
Aktuell produziert das Skript einen Fehler, den ich nicht mehr ausmerzen konnte. 

Im Terminal kann task.py folgendermaßen verwendet werden: 
```python task.py -dir './../data/```.

Durch den Aufruf wird ein Modell traininert und abgespeichert. Es kann anschließend mit einem einfachen predict-Befehl zur Vorhersage genutzt werden. In der akutellen Form ist kein Preprocessing der zur Vorhersage benötigten Features nötig. 
 


