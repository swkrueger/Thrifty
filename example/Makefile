all: rx.clock_model rx.toads_analysis

TOAD_FILES 		:= $(patsubst cards/%.card, toad/%.toad, $(wildcard cards/*.card))

toad/%.toad: cards/%.card detector.cfg
	@mkdir -p toads
	thrifty detect $< -o $@

rx.toads: $(TOAD_FILES) detector.cfg
	thrifty integrate $(TOAD_FILES) -o rx.toads

rx.toads_analysis: rx.toads detector.cfg
	thrifty analyze_toads rx.toads > rx.toads_analysis

rx.match: rx.toads detector.cfg
	thrifty match rx.toads -o rx.match

rx.clock_model: rx.toads rx.match detector.cfg
	thrifty clock_sync rx.toads rx.match
	touch rx.clock_model

detect: rx.toads
integrate: detect
analyze_toads: rx.toads_analysis
match: rx.match
clock_sync: rx.clock_model
