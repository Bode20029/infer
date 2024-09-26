from roboflow import Roboflow

rf = Roboflow(api_key="o3boI5caOanTQXiFIH6j")
project = rf.workspace("UNIVERSITY").project("ev-car-vixid")
model = project.version(3, local="http://localhost:9001/").model

prediction = model.predict("YOUR_IMAGE.jpg", confidence=40, overlap=30)
## get predictions on hosted images
#prediction = model.predict("YOUR_IMAGE.jpg", hosted=True)
print(prediction.json())