all
rule 'MD007', :indent => 4
rule 'MD013', :line_length => 500
rule 'MD029', :style => :ordered
rule "MD030", :ul_multi => 3, :ol_multi => 2, :ol_single => 2, :ul_single => 3
rule 'MD024', :allow_different_nesting => true
rule 'MD026', :punctuation => '.,;:!'
exclude_rule 'MD034'
