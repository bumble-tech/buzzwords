if [[ -z "$1" ]]; then
	echo "You need to give an image - usage is ${0} 'image' (e.g. ${0} europe-west4-docker.pkg.dev/ds-team-sandbox/buzzwords/buzzwords-base:1.1)"
	exit 1
fi

set -euo pipefail

sudo docker build -f $(dirname ${BASH_SOURCE[0]})/Dockerfile -t test-buzzwords $(dirname ${BASH_SOURCE[0]}) --build-arg IMAGE=$1

sudo docker run test-buzzwords