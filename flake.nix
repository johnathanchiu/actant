{
  description = "Actant development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs, ... }: {
    # Rift VM profile: tools on the dev user's path in every rift box.
    fixed-labs.rift.x86_64-linux =
      let pkgs = nixpkgs.legacyPackages.x86_64-linux; in {
        # Python 3.11 matches `python_version` in the justfile; uv and just
        # run the justfile's recipes.
        packages = with pkgs; [ python311 uv just ];
        # uv downloads generic Linux interpreters that do not run on NixOS;
        # keep it on the nix-provided one.
        env.UV_PYTHON_PREFERENCE = "only-system";
      };
  };
}
