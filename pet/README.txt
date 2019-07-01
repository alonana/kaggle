
In this competition you will predict the speed at which a pet is adopted, based on the pet’s listing on PetFinder.
Sometimes a profile represents a group of pets.
In this case, the speed of adoption is determined by the speed at which all of the pets are adopted.
The data included text, tabular, and image data. See below for details.
This is a Kernels-only competition.
At the end of the competition, test data will be replaced in their entirety with new data of approximately the same size,
and your kernels will be rerun on the new data.

File descriptions
train.csv - Tabular/text data for the training set
test.csv - Tabular/text data for the test set
sample_submission.csv - A sample submission file in the correct format
breed_labels.csv - Contains Type, and BreedName for each BreedID. Type 1 is dog, 2 is cat.
color_labels.csv - Contains ColorName for each ColorID
state_labels.csv - Contains StateName for each StateID

Data Fields
PetID - Unique hash ID of pet profile
AdoptionSpeed - Categorical speed of adoption. Lower is faster. This is the value to predict. See below section for more info.
Type - Type of animal (1 = Dog, 2 = Cat)
Name - Name of pet (Empty if not named)
Age - Age of pet when listed, in months
Breed1 - Primary breed of pet (Refer to BreedLabels dictionary)
Breed2 - Secondary breed of pet, if pet is of mixed breed (Refer to BreedLabels dictionary)
Gender - Gender of pet (1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets)
Color1 - Color 1 of pet (Refer to ColorLabels dictionary)
Color2 - Color 2 of pet (Refer to ColorLabels dictionary)
Color3 - Color 3 of pet (Refer to ColorLabels dictionary)
MaturitySize - Size at maturity (1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large, 0 = Not Specified)
FurLength - Fur length (1 = Short, 2 = Medium, 3 = Long, 0 = Not Specified)
Vaccinated - Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)
Dewormed - Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)
Sterilized - Pet has been spayed / neutered (1 = Yes, 2 = No, 3 = Not Sure)
Health - Health Condition (1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)
Quantity - Number of pets represented in profile
Fee - Adoption fee (0 = Free)
State - State location in Malaysia (Refer to StateLabels dictionary)
RescuerID - Unique hash ID of rescuer
VideoAmt - Total uploaded videos for this pet
PhotoAmt - Total uploaded photos for this pet
Description - Profile write-up for this pet. The primary language used is English, with some in Malay or Chinese.

AdoptionSpeed
Contestants are required to predict this value. The value is determined by how quickly, if at all, a pet is adopted.
The values are determined in the following way:
0 - Pet was adopted on the same day as it was listed.
1 - Pet was adopted between 1 and 7 days (1st week) after being listed.
2 - Pet was adopted between 8 and 30 days (1st month) after being listed.
3 - Pet was adopted between 31 and 90 days (2nd & 3rd month) after being listed.
4 - No adoption after 100 days of being listed. (There are no pets in this dataset that waited between 90 and 100 days).

Images
For pets that have photos, they will be named in the format of PetID-ImageNumber.jpg.
Image 1 is the profile (default) photo set for the pet.
For privacy purposes, faces, phone numbers and emails have been masked.

Image Metadata
We have run the images through Google's Vision API, providing analysis on Face Annotation,
Label Annotation, Text Annotation and Image Properties.
You may optionally utilize this supplementary information for your image analysis.
File name format is PetID-ImageNumber.json.
Some properties will not exist in JSON file if not present, i.e. Face Annotation.
Text Annotation has been simplified to just 1 entry of the entire text description
(instead of the detailed JSON result broken down by individual characters and words).
Phone numbers and emails are already anonymized in Text Annotation.
Google Vision API reference: https://cloud.google.com/vision/docs/reference/rest/v1/images/annotate

Sentiment Data
We have run each pet profile's description through Google's Natural Language API,
providing analysis on sentiment and key entities.
You may optionally utilize this supplementary information for your pet description analysis.
There are some descriptions that the API could not analyze. As such, there are fewer sentiment files than there are rows in the dataset.
File name format is PetID.json.
Google Natural Language API reference: https://cloud.google.com/natural-language/docs/basics

What will change in the 2nd stage of the competition?
In the second stage of the competition, we will re-run your selected Kernels.
The following files will be swapped with new data:

test.zip including test.csv and sample_submission.csv
test_images.zip
test_metadata.zip
test_sentiment.zip
In stage 2, all data will be replaced with approximately the same amount of different data.
The stage 1 test data will not be available when kernels are rerun in stage 2.