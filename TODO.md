When running the Docker file directly, the generation succeeds showing GPU Memory Usage of 5186MiB. I can hear my GPU fan running during the 4-minute generation.

When I try running the same Docker file in Lilypad, the first time I run it, it takes about 20 minutes to download, then I get an OOM error.

The second time I run it, it fails after two minutes with an OOM error. I don't ever hear my GPU fan running.