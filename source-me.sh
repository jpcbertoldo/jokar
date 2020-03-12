DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
HISTWORDS="${DIR}/histwords"
#export HISTWORDSPATH="${HISTWORDS}; ${HISTWORDS}/representations; ${HISTWORDS}/coha"
export HISTWORDSPATH="$(find ${HISTWORDS}/ -maxdepth 1 -type d | sed '/\/\./d' | tr '\n' ':' | sed 's/:$//')"

echo "To run histwords scripts, used 'PYTHONPATH=\$HISTWORDSPATH python /path/to/script.py'"
echo ""
echo "\$HISTWORDSPATH: ${HISTWORDSPATH}"
echo ""
