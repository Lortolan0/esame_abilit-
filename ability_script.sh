

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

URL="https://adlibitum.oats.inaf.it/monaco/etc/perAbInf.tgz"

DEST_DIR="$SCRIPT_DIR"

wget -P "$DEST_DIR" -r -nd "$URL"

if [ $? -eq 0 ]; then
	echo "Download completato con successo."

	cd "$DEST_DIR"

	for file in *.tgz; do
		tar -zxvf "$file" && rm "$file"
	done



	FINAL_DEST="$SCRIPT_DIR"

	mv * "$FINAL_DEST"


	echo "File estratti copiati in $FINAL_DEST."

else
	echo "Si è verificato un errore durante il download."
fi
