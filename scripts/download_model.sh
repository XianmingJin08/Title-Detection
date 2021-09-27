DIR='./models/bert/'
URL='https://drive.google.com/uc?id=1m2kVhguW62gfYh-Pndf5tdVqjSUfaTvJ'
echo "Downloading pre-trained models..."
mkdir -p $DIR
cd $DIR
gdown $URL
cd ../../
DIR='./models/bertSequence/'
URL='https://drive.google.com/uc?id=1X27orNRq6fHc5gTcWJqxa-0QA20Hl7Lp'
mkdir -p $DIR
cd $DIR
gdown $URL
cd ../../
echo "Download success."
