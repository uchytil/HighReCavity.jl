using Documenter, HighReCavity

makedocs(
    sitename = "HighReCavity.jl",
    modules = [HighReCavity],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true", mathengine = Documenter.MathJax3(),
                             edit_link = "main", canonical = "https://uchytil.github.io/HighReCavity.jl/stable/"),
    pages = [
        "Home" => "index.md",
        "Numerical method" => "method.md",
        "Usage" => "usage.md",
        "API" => "api.md",
    ],
)

deploydocs(repo = "github.com/uchytil/HighReCavity.jl.git", devbranch = "main", push_preview = false)
