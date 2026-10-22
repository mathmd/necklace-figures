{
  description = "Figures 2-5 of 'Stationary bistable fronts on a periodic necklace graph'";

  # The nixpkgs revision that produced the figures in the paper; flake.lock records its hash.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/e2587caef70cea85dd97d7daab492899902dbf5d";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs: {
        # Python with NumPy, SciPy, Matplotlib and Shapely, a C++ compiler (from mkShell),
        # and LaTeX for the figure text.
        default = pkgs.mkShell {
          packages = [
            (pkgs.python3.withPackages (ps: with ps; [ numpy scipy matplotlib shapely ]))
            (pkgs.texliveMedium.withPackages (ps: with ps; [ type1cm cm-super underscore ]))
          ];
        };
      });
    };
}
