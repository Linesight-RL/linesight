for (let b of document.getElementsByClassName("navbutton")) {
  b.onclick = function(e: Event) {
    if (!e.target) {
      return;
    }
    for (let p of document.getElementsByClassName("page")) {
      p.classList.remove("active");
    }
    for (let b of document.getElementsByClassName("navbutton")) {
      b.classList.remove("active");
    }
    e.target.classList.add("active");
    document.getElementById("_" + e.target.id)?.classList.add("active");
  };
}

function log(msg: string) {
  console.log(msg);
  UiLog.get().componentUpdate({message: msg});
}

function update() {
  if (Object.entries(loader.database.values).length > 1) {
    for (let b of document.getElementsByClassName("hidden-games")) {
      b.classList.remove("hidden-games");
    }
  }
  if (!UiGames.get().game) {
    let game = loader.database.get_game(loader.config.default_game);
    UiGames.get().set(game);
  }
  UiGames.get().componentUpdate();
  UiProducts.get().componentUpdate();
  UiDescription.get().componentUpdate();
  //UiConfig.get().componentUpdate();
}

// i think that this is the most reliable method to schedule an async funtion to execute after load
let debounce = true;
document.addEventListener("load", async () => {
  if (debounce) {
    debounce = false;
    log("Loading database");
    await loader.database.async_load();
    update();
    log("Fetching products from servers");
    let err = await loader.async_fetch();
    log("Saving database");
    await loader.database.async_save(); // we need that if we want to be able to install mods that we fetched
    if (err) {
      log("Failed to fetch: " + err);
    } else {
      log("Ready");
    }
    update();
    setTimeout(() => {debounce = true}, 1000);
  }
});

document.on('click', ".bar-button[role=bar-add]", async function() {
  let id = 1;
  let game = UiGames.get().game;
  if (game) {
    while (game.get_profile("New Profile " + id)) {
      id += 1;
    }
    let profile = game.add_profile("New Profile " + id);
    if (profile) {
      profile.program.id = game.get_name();
      UiProfiles.get().set(profile);
      UiProfiles.get().componentUpdate();
      await profile.async_save();
    }
    UiDescription.get().product = false;
    UiDescription.get().componentUpdate();
  }
});

export class UiGames extends Element {
  game: Game|null = null;

  static get(): UiGames {
    return document.getElementById("_games") as UiGames;
  }

  render() {
    return <span>{Object.entries(loader.database.values).map(([game_name, game], idx) => this.UxGame(game_name, game, this.game ? this.game.get_name() == game.get_name() : idx == 0))}</span>;
  }
  UxGame(name: string, game: Game, selected: boolean = false) {
    if (selected) {
      this.set(game);
    }
    return <button class={selected ? "panel selected selectable" : "panel"} id={"_game_" + name} title={name} onclick={() => {this.set(game)}}>
        <div class="panel-header">
          <div class="panel-caption"><b>{name}</b></div>
        </div>
      </button>;
  }
  set(game: Game|null = null) {
    let same: boolean = game?.get_name() == this.game?.get_name();
    this.game = game;

    if (!same) {
      // manually handle selection to avoid rerender
      for (let p of this.$$(".panel.selected")) {
        p.classList.remove("selected");
      }
      if (game) {
        document.getElementById("_game_" + game.get_name())?.classList.add("selected");
        loader.config.default_game = game.get_name();
        
        if (Object.entries(game.profiles.values).length == 0) {
          let profile = game.add_profile("default");
          if (profile) {
            profile.program.id = game.get_name();
          }
        }
        UiProfiles.get().set(game.get_profile(loader.config.default_profiles[game.get_name()]));
        UiProfiles.get().componentUpdate();
      }
    }
  }
}

export class UiProfiles extends Element {
  profile: Profile|null = null;

  static get(): UiProfiles {
    return document.getElementById("_profiles") as UiProfiles;
  }

  render() {
    let game = UiGames.get().game;
    if (!game) {
      return <span/>;
    }
    return <span>{Object.entries(game.profiles.values).map(([profile_name, profile], idx) => this.UxProfile(profile_name, profile, this.profile ? this.profile.get_name() == profile.get_name() : idx == 0))}</span>;
  }
  UxProfile(name: string, prof: Profile, selected: boolean = false) {
    if (selected) {
      this.set(prof);
    }
    let fixed = prof.get_name();
    return <button class={selected ? "panel selected" : "panel"} id={"_prof_" + fixed} title={fixed}
      onclick={() => {this.set(prof)}}><div class="panel-header">
        <button class="panel-button" role="panel-run" title="Play" onclick={async () => {
          log("Installing newest products for profile '" + fixed + "', please wait");
          let err = await prof.async_update();
          log("Launching profile '" + fixed + "'");
          await prof.prepare()?.async_start();
          if (err) {
            log("Failed to install newest products for profile '" + fixed + "': " + err);
          } else {
            log("Ready");
          }
          UiDescription.get().componentUpdate();
        }}/>
        <div class="panel-caption">{fixed}</div>
        <div class="panel-buttons">
          <button class="panel-button" role="panel-remove" title="Remove" onclick={() => {
            this.profile?.profiles?.remove_profile(prof.get_name());
            if (this.profile?.get_name() == prof.get_name()) {
              this.componentUpdate({profile: null});
            } else this.componentUpdate();
          }}/>
        </div>
      </div></button>;
  }
  set(profile: Profile|null = null) {
    this.profile = profile;
    for (let p of this.$$(".panel.selected")) {
      p.classList.remove("selected");
    }
    for (let p of UiProducts.get().$$(".panel-button.filled")) {
      p.classList.remove("filled");
    }
    if (profile) {
      document.getElementById("_prof_" + profile.get_name())?.classList.add("selected");
      document.getElementById("_check_prod_" + profile.program.id)?.classList.add("filled");
      if (profile.mods) {
        for (let mod of profile.mods) {
          document.getElementById("_check_prod_" + mod.id)?.classList.add("filled");
        }
      }
      loader.config.set_default_profile(loader.config.default_game, profile.get_name());
    }
    UiProducts.get().componentUpdate({product: null});
    UiDescription.get().product = false;
    UiDescription.get().componentUpdate();
  }
}

export class UiProducts extends Element {
  product: Product|null = null;

  static get(): UiProducts {
    return document.getElementById("_products") as UiProducts;
  }

  render() {
    let profile = UiProfiles.get().profile;
    let game = UiGames.get().game;
    if (!game) {
      return <span/>;
    }
    let products = Object.entries(game.products.values);
    let programs = products.filter(([product_name, product]) => product.description.type == "program");
    let mods = products.filter(([product_name, product]) => product.description.type == "modification");
    ;
    return <span>{programs.length ? <div class="separator"><span>Programs</span></div> : []}
      {programs.map(([product_name, product]) => this.UxProd(product_name, product, this.product?.get_name() == product_name, profile?.program.id == product_name))}
      {mods.length ? <div class="separator"><span>Mods</span></div> : []}
      {mods.map(([product_name, product]) => this.UxProd(product_name, product, this.product?.get_name() == product_name, profile?.has_mod(product_name)))}</span>;
  }
  UxProd(name: string, prod: Product, selected: boolean = false, filled: boolean = false) {
    let clazz = "panel";
    if (selected) {
      clazz += " selected";
    }
    return <button class={clazz} id={"_prod_" + name} title={name}
      onclick={() => {
        this.set(prod);
      }}><div class="panel-header">
        <button class={filled ? "panel-button filled" : "panel-button"} role="panel-select" title="Select" id={"_check_prod_" + name} onclick={async () => {
          let profile: Profile|null = UiProfiles.get().profile;
          if (profile) {
            if (prod.description.type == "program") {
              profile.program.id = prod.get_name();
              profile.program.version = null;
              //profile.program.prerelease = null;
            } else {
              profile.toggle_mod(prod.get_name());
            }
            this.componentUpdate();
            UiDescription.get().componentUpdate();
            await profile.async_save();
          }
        }}/>
        <div class="panel-caption">{name == UiGames.get().game?.get_name() ? <b>{name}</b> : name}</div>
        {prod.description.icon_link ? <img class="panel-image" src={prod.description.icon_link}/> : []}
      </div></button>;
  }
  set(product: Product|null = null) {
    this.product = product;
    for (let p of this.$$(".panel.selected")) {
      p.classList.remove("selected");
    }
    if (product) {
      document.getElementById("_prod_" + product.get_name())?.classList.add("selected");
      UiDescription.get().product = true;
      UiDescription.get().componentUpdate();
    }
  }
}


export class UiDescription extends Element {
  product: boolean = false;

  version: ProductVersion|null = null;

  static get(): UiDescription {
    return document.getElementById("_description") as UiDescription;
  }

  render() {
    let profile = UiProfiles.get().profile;
    let product = UiProducts.get().product;
    if (!this.product && profile) {
      return <div class="description">
        <button class="panel">
          <div class="panel-header">
            <div class="panel-image" role="panel-edit" title="Rename"/>
            <div class="panel-caption"><input class="panel-input" type="text" title="Edit name" id="_edit_profile" value={profile.get_name()}
              onkeydown={(e: Event) => {
                if (e.keyCode == 257) { // KB_ENTER
                  profile.profiles?.rename_profile(profile.get_name(), document.getElementById("_edit_profile")?.value);
                  UiProfiles.get().componentUpdate();
                }
              }}
              onfocusout={() => {
                profile.profiles?.rename_profile(profile.get_name(), document.getElementById("_edit_profile")?.value);
                UiProfiles.get().componentUpdate();
              }}/>
            </div>
          </div>
        </button>
        <hr/>
        <button class="panel highlighted" id="_but_play" onclick={async () => {
          let fixed = profile.get_name();
          log("Installing newest products for profile '" + fixed + "', please wait");
          let err = await profile.async_update();
          log("Launching profile '" + fixed + "'");
          await profile.prepare()?.async_start();
          if (err) {
            log("Failed to install newest products for profile '" + fixed + "': " + err);
          } else {
            log("Ready");
          }
          UiDescription.get().componentUpdate();
        }}><div class="panel-header">
            <div class="panel-image" role="panel-run"/>
            <div class="panel-caption">Play</div>
            <div class="panel-buttons">
              <button class="panel-button" role="panel-shortcut" title="Make a shortcut on desktop" onclick={async () => {
                log("Making shortcut on desktop for profile '" + profile.get_name() + "'");
                profile.make_shortcut_on_desktop();
                log("Ready");
              }}/>
            </div>
          </div>
        </button>
        <div class="separator hr"><span>Program</span></div>
        <button class="panel" title="Args"><div class="panel-header"><div class="panel-caption"><b>Args:</b>
          <input type="text" class="panel-input" id="_args"
            oninput={() => {
              let value: string = document.getElementById("_args")?.value;
              if (value && value.length) {
                profile.args = value;
              } else {
                profile.args = null;
              }
            }}
            onkeydown={async (e: Event) => {
              if (e.keyCode == 257) { // KB_ENTER
                await profile.async_save();
              }
            }}
            onfocusout={async () => {
              await profile.async_save();
            }}
          >{profile.args}</input></div></div>
        </button>
        {this.UxSelector(profile, profile.program, false)}
        <div class="separator hr"><span>Mods</span></div>
        {profile.mods?.map((sel) => this.UxSelector(profile, sel))}
      </div>;
    } else if (this.product && product) {
      return <div class="description">
        <button class="panel" title="Info" onclick={() => {
          document.getElementById("_desc_prod")?.classList.toggle("collapsed");
        }}>
          <div class="panel-header"><div class="panel-caption"><b>Name:</b>{product.description.name}</div></div>
          <div class="panel-header"><div class="panel-caption"><b>Author:</b>{product.description.author}</div></div>
          {product.description.homepage? <div class="panel-header"><div class="panel-caption"><b>Homepage:</b><a href={product.description.homepage}>{product.description.homepage}</a></div></div> : []}
          {product.description.description? <div class="panel-description" id="_desc_prod"><b>Description:</b>{product.description.description}</div> : []}
        </button>
        { product.description.external ?
          ([<div class="separator hr"><span>Integration</span></div>, <button class="panel" title="Integration" id="_but_integration" onclick={() => {
            document.getElementById("_desc_integration")?.classList.toggle("collapsed");
          }}>
            <div class="panel-header">
              <div class="panel-caption"><b>Integration</b></div>
              <div class="panel-buttons">{product.integration_installed() ?
                <button class="panel-button" role="panel-remove" title="Uninstall" onclick={async () => {
                  log("Uninstalling integration for " + product.get_name())
                  product.integration_uninstall();
                  this.componentUpdate();
                  log("Ready");
                }}/> :
                <button class="panel-button" role="panel-install" title="Install" onclick={async () => {
                  log("Installing integration for " + product.get_name())
                  product.integration_install();
                  this.componentUpdate();
                  log("Ready");
                }}/> }
              </div>
            </div>
            <div class={product.integration_installed() ? "panel-description collapsed" : "panel-description"} id="_desc_integration">Integration replaces the original game to automatically launch modloader with 'default' profile.
You must launch the game at least once before installing integration.
Admin privileges might be required.</div>
          </button>]) : []
        }
        <div class="separator hr"><span>Versions</span></div>
        {this.UxVersions(product)}
      </div>;
    } else return <div/>;
  }
  UxSelector(profile: Profile, selector: Selector|null, is_mod: boolean = true) {
    if (selector) {
      return <button class="panel" title={selector.id} id={"_sel_" + selector.id}>
          <div class="panel-header">
            <button class="panel-caption" onclick={() => {
              document.getElementById("_desc_sel_" + selector.id)?.classList.toggle("collapsed");
            }}><b>Id:</b>{selector.id}</button>
            {is_mod ? <div class="panel-buttons">
              <button class="panel-button" role="panel-remove" title="Remove" onclick={async () => {
                profile.remove_mod(selector.id);
                UiProfiles.get().componentUpdate();
                this.componentUpdate();
                await profile.async_save();
              }}/>
            </div> : []}
          </div>
          <div class={selector.version ? "panel-description" : "panel-description collapsed"} id={"_desc_sel_" + selector.id}><b>Version:</b>
            <input type="text" class="panel-input" id={"_sel_ver_" + selector.id} value={selector.version}
              oninput={() => {
                let value: string = document.getElementById("_sel_ver_" + selector.id)?.value;
                if (value && value.length) {
                  selector.version = value;
                } else {
                  selector.version = null;
                }
              }}
              onkeydown={async (e: Event) => {
                if (e.keyCode == 257) { // KB_ENTER
                  await profile.async_save();
                }
              }}
              onfocusout={async () => {
                await profile.async_save();
              }}
            />
          </div>
        </button>;
    }
    return [];
  }
  UxVersions(prod: Product) {
    return Object.entries(prod.values).map(([version_name, version], idx) => {
      let profile = UiProfiles.get().profile;
      let filled: boolean = false;
      if (profile) {
        if (prod.description.type == "program") {
          filled = profile.program.version == version_name;
        } else {
          filled = profile.get_mod(prod.get_name())?.version == version_name;
        }
      }
      return this.UxVersion(version_name, version, idx != 0, filled);
    });
  }
  UxVersion(name: string, version: ProductVersion, collapsed: boolean = false, filled: boolean = false) {
    return <button class="panel" id={"_ver_" + name} onclick={() => {
        document.getElementById("_desc_ver_" + name)?.classList.toggle("collapsed");
      }} title={name}>
        <div class="panel-header">
          {(!version.is_installed() && !version.is_installable()) ?
            <div class="panel-image"/> :
            <button class={filled ? "panel-button filled" : "panel-button"} role="panel-select" title="Select" id={"_check_ver_" + name} onclick={async () => {
              let profile: Profile|null = UiProfiles.get().profile;
              if (profile) {
                let prod = version.product;
                if (prod) {
                  for (let b of this.getElementsByClassName("filled")) {
                    b.classList.remove("filled");
                  }
                  if (prod.description.type == "program") {
                    profile.program.id = prod.get_name();
                    if (profile.program.version == name) {
                      profile.program.version = null;
                    } else {
                      profile.program.version = name;
                      document.getElementById("_check_ver_" + name)?.classList.add("filled");
                    }
                  } else {
                    let mod = profile.get_mod(prod.get_name());
                    if (mod) {
                      if (mod.version == name) {
                        mod.version = null;
                      } else {
                        mod.version = name;
                        document.getElementById("_check_ver_" + name)?.classList.add("filled");
                      }
                    } else {
                      mod = profile.add_mod(prod.get_name());
                      mod.version = name;
                      document.getElementById("_check_ver_" + name)?.classList.add("filled");
                    }
                  }
                }
                UiProducts.get().componentUpdate();
                await profile.async_save();
              }
            }}/>
          }
        <div class="panel-caption"><b>{name}</b></div>
        <div class="panel-buttons">{
          version.is_installed()? 
            <button class="panel-button" role="panel-remove" title="Uninstall" onclick={async () => {
              log("Uninstalling " + version.product?.get_name() + " " + name);
              if (!await version.async_uninstall()) {
                log("Failed to uninstall " + version.product?.get_name() + " " + name);
              } else log("Ready");
              this.componentUpdate();
            }}/> : (
          version.is_installable()? 
            <button class="panel-button" role="panel-install" title="Install" onclick={async () => {
              log("Installing " + version.product?.get_name() + " " + name);
              let err = await version.async_install();
              if (err) {
                log("Failed to install " + version.product?.get_name() + " " + name + ": " + err);
              } else log("Ready");
              this.componentUpdate();
            }}/>
          :[])}</div>
      </div>
      {version.description.changelog? <div class={collapsed ? "panel-description collapsed" : "panel-description"} id={"_desc_ver_" + name}>{version.description.changelog}</div> : []}
    </button>;
  }
}

export class UiConfig extends Element {
  componentDidMount() {
    this.content(this.render());
  }
  static get(): UiConfig {
    return document.getElementById("_config") as UiConfig;
  }
  render() {
    // protocol install/uninstall
    // fetch: (tester mode)
    //ui: ConfigUi;
    //readonly masterserver: string;
    // i forgot database is readonly
    //<button class="panel" title="Database"><div class="panel-header"><div class="panel-caption"><b>Database:</b>
    //  <input type="text" class="panel-input" id="_input_database"
    //    oninput={() => {loader.config.database = document.getElementById("_input_database")?.value}}
    //    onkeydown={async (e: Event) => { if (e.keyCode == 257) {await loader.config.async_save()}}}
    //    onfocusout={async () => {await loader.config.async_save()}}
    //  >{loader.config.database}</input></div></div>
    //</button>
    return <span>
      <button class="panel" title="Portable" onclick={() => {
        document.getElementById("_desc_portable")!.classList.toggle("collapsed");
      }}>
        <div class="panel-header">
          <button class={loader.config.portable ? "panel-button filled" : "panel-button"} role="panel-select" id="_check_portable" onclick={async () => {
            loader.config.portable = document.getElementById("_check_portable")!.classList.toggle("filled");
            await loader.config.async_save();
          }}/>
          <div class="panel-caption">Portable</div>
        </div>
        <div class="panel-description" id="_desc_portable">When it is portable, it will not check protocol integrity every launch.</div>
      </button>
      <button class="panel" title="Auto Update" onclick={() => {
        document.getElementById("_desc_autoupdate")!.classList.toggle("collapsed");
      }}>
        <div class="panel-header">
          <button class={loader.config.auto_update != "false" ? "panel-button filled" : "panel-button"} role="panel-select" id="_check_autoupdate" onclick={async () => {
            if (document.getElementById("_check_autoupdate")!.classList.toggle("filled")) {
              loader.config.auto_update = "after";
            } else {
              loader.config.auto_update = "false";
            }
            await loader.config.async_save();
          }}/>
          <div class="panel-caption">Auto Update</div>
        </div>
        <div class="panel-description" id="_desc_autoupdate">Automatically update the game or mods after launching a profile using a link or integration.
It will download the update in the background and it will be available the next time you launch it</div>
      </button>
      <button class="panel" title="Auto Upgrade" onclick={() => {
        document.getElementById("_desc_autoupgrade")!.classList.toggle("collapsed");
      }}>
        <div class="panel-header">
          <button class={loader.config.auto_upgrade != "false" ? "panel-button filled" : "panel-button"} role="panel-select" id="_check_autoupgrade" onclick={async () => {
            if (document.getElementById("_check_autoupgrade")!.classList.toggle("filled")) {
              loader.config.auto_upgrade = "after";
            } else {
              loader.config.auto_upgrade = "false";
            }
            await loader.config.async_save();
          }}/>
          <div class="panel-caption">Auto Upgrade</div>
        </div>
        <div class="panel-description" id="_desc_autoupgrade">Automatically update the modloader after closing it</div>
      </button>
    </span>;
  }
}

export class UiLog extends Element {
  message: string = "Hello!";

  static get(): UiLog {
    return document.getElementById("_log") as UiLog;
  }

  componentDidMount() {
    this.content(this.render());
  }
  render() {
    return <span>{this.message}</span>;
  }
}

export class UiVersion extends Element {
  componentDidMount() {
    this.content(this.render());
  }
  render() {
    return <span>{loader.config.version}</span>;
  }
}

