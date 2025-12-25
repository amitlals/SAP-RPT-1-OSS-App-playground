#!/bin/bash

# Set TabPFN token if provided
if [ -n "$TABPFN_TOKEN" ]; then
    python -c "from tabpfn_client import set_access_token; set_access_token('$TABPFN_TOKEN')"
    echo "TabPFN token configured"
fi

# Start supervisor (manages both API and Streamlit)
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
