# Farbe für den Prompt definieren
PROMPT_COLOR='\[\033[32m\]'  # Beispiel: Grün, kann auf eine andere Farbe geändert werden

# Funktion zur Verkürzung des Pfads auf den untersten Ordner
shorten_path() {
    echo "${PWD##*/}"
}

# Wenn es einen alten Prompt gibt, speichere ihn
if [ -n "${_OLD_VIRTUAL_PS1:-}" ] ; then
    PS1="${_OLD_VIRTUAL_PS1:-}"
else
    _OLD_VIRTUAL_PS1="$PS1"
fi

# Setze den neuen farbigen Prompt
PS1="${PROMPT_COLOR}\$(date +'%y%m%d-%H%M')$\[\033[0m\]\u@\h:\$(shorten_path)\\"
export PS1
