source activate buzzwords2

# Install npdoc if it hasn't already been
if [[ -z "$(pip3 list | grep -F npdoc-to-md)" ]]; then
	python3 -m pip install npdoc-to-md==1.1
fi

echo "Saving documentation to API_docs.md"

python3 print.py > 'API_docs.md'

echo "Documentation saved to API_docs.md"