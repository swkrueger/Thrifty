all: rx.clock_model rx.toads_analysis

TOAD_FILES 		:= $(patsubst cards/%.card, toad/%.toad, $(wildcard cards/*.card))

toad/%.toad: cards/%.card thrifty.cfg
	@mkdir -p toads
	thrifty detect $< -o $@

rx.toads: $(TOAD_FILES) thrifty.cfg
	thrifty integrate $(TOAD_FILES) -o rx.toads

rx.toads_analysis: rx.toads thrifty.cfg
	thrifty analyze_toads rx.toads > rx.toads_analysis

rx.match: rx.toads thrifty.cfg
	thrifty match rx.toads -o rx.match

rx.clock_model: rx.toads rx.match thrifty.cfg
	thrifty clock_sync rx.toads rx.match
	touch rx.clock_model

detect: rx.toads
integrate: detect
analyze_toads: rx.toads_analysis
match: rx.match
clock_sync: rx.clock_model
