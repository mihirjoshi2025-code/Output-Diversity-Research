-- Turn fenced divs like ::: {.exercise title="2.1 Bayes accuracy"} into tcolorbox environments.
local labels = {
  exercise = "Exercise",
  concept = "Key idea",
  pitfall = "Pitfall",
  checkpoint = "Checkpoint",
  repo = "In the repo",
  mentor = "Ask your mentor",
}

function Div(el)
  for _, class in ipairs(el.classes) do
    local label = labels[class]
    if label then
      local title = el.attributes.title
      local full = title and (label .. ": " .. title) or label
      local blocks = { pandoc.RawBlock("latex", "\\begin{" .. class .. "}{" .. full .. "}") }
      for _, b in ipairs(el.content) do
        table.insert(blocks, b)
      end
      table.insert(blocks, pandoc.RawBlock("latex", "\\end{" .. class .. "}"))
      return blocks
    end
  end
end
