type Err = string?;
declare class YamlNode {
    //import(str: string): boolean;
    export_yaml(): string;
    export_json(): string;
}
declare class YamlFs extends YamlNode {
    readonly path: string;

    get_name(): string;
    load(): boolean;
    save(): void;
    async async_load(): boolean;
    async async_save(): void;
    //load_for(name: string): boolean;
    //save_for(name: string): void;
    //load_as(path: string, update: boolean): boolean;
    //save_as(path: string): void;
    backup(): void;
    backup_restore(): boolean;
    backup_save(): void;
    backup_load(): boolean;
    //copy(name: string): boolean;
    //rename(name: string): boolean;
    //remove(): void;
    exists(): boolean;
    is_backup(): boolean;
    has_backup(): boolean;
}
declare class YamlFile extends YamlFs {}
declare class YamlDir extends YamlFs {}
declare class YamlMulti<Type = YamlFs> extends YamlDir {
    values : { [name: string]: Type };
}
declare class YamlRecursive<Type = YamlFs> extends YamlMulti<Type> {}

declare class ProcessInfo {
    readonly exe: string;
    readonly cwd: string?;
    cmd: string;
    title: string;
    readonly dlls: string[];
    readonly envpaths: string[];
    debug: boolean;
    readonly caller_pid: int;

    start(): ProcessHandler?;
    async async_start(): ProcessHandler?;
}

declare class ConfigUi extends YamlNode {
    style: string;
    layout: string;
}
declare class Config extends YamlFile {
    readonly version: string;
    portable: boolean;
    readonly database: string;
    auto_update: "false"|"before"|"after";
    auto_upgrade: "false"|"before"|"after";
    ui: ConfigUi;
    default_game: string;
    readonly default_profiles: { [name: string]: string };
    readonly masterserver: string;
    //readonly servers: { [name: string]: string }; // FIXME: servers.servers

    set_default_profile(game: string, profile: string): void;
}

declare class Selector extends YamlNode {
    id: string;
    version: string?;
    prerelease: boolean?;
}
declare class Selection extends YamlNode {
}

declare class ProductVersionDescription extends YamlFile {
    productversion: ProductVersion?;

    readonly executable: string;
    readonly load_method: "runtime" | "import" | null;
    readonly priority: number;
    readonly dependencies: Selector[]?;
    readonly download_link: string?;
    readonly hash_sha256: string?;
    readonly password: string?;
    readonly changelog: string?;
}
declare class ProductDescription extends YamlFile {
    product: Product?;

    readonly name: string;
    readonly author: string;
    readonly type: "modification"|"program";
    readonly external: boolean?;
    readonly homepage: string?;
    readonly description: string?;
    readonly icon_link: string?;
}
declare class Settings extends YamlFile {
    product: Product?;
    install: string?;
}

declare class Profile extends YamlFile {
    profiles: Profiles?;
    program: Selector;
    args: string?;
    mods: Selector[]?;
    
    update(): string;
    async async_update(): string;
    prepare(): ProcessInfo?;
    
    get_mod(product: string): Selector?;
    has_mod(product: string): boolean;
    add_mod(product: string): Selector;
    remove_mod(product: string): void;
    toggle_mod(product: string): void;

    make_shortcut_on_desktop(): boolean;

    // 0=any, 1=installable, 2=installed
    //resolve(installed: number): Selection[]; // WARNING! output is temporary, dont reference the items!
}

declare class ProductVersion extends YamlDir {
    product: Product?;
    description: ProductVersionDescription;

    is_installed(): boolean;
    is_installable(): boolean;
    async async_install(): string;
    async async_reinstall(): string;
    async async_uninstall(): boolean;
}

declare class Product extends YamlMulti<ProductVersion> {
    products: Products?;
    settings: Settings?;
    description: ProductDescription;

    get_version(version: string): ProductVersion?;
    external_install_valid(): boolean;
    integration_installable(): boolean;
    integration_installed(): boolean;
    integration_install(): boolean;
    integration_uninstall(): boolean;
}

declare class Products extends YamlMulti<Product> {
    game: Game?;
}
declare class Profiles extends YamlRecursive<Profile> {
    game: Game?;
    add_profile(name: string): Profile?;
    get_profile(name: string): Profile?;
    rename_profile(from: string, to: string): boolean;
    remove_profile(name: string): boolean;
}

declare class Game extends YamlDir {
    database: Database?;
    products: Products;
    profiles: Profiles;

    get_product(product: string): Product?;
    get_version(product: string, version: string): ProductVersion?;
    add_profile(name: string): Profile?;
    get_profile(name: string): Profile?;
    rename_profile(from: string, to: string): boolean;
    remove_profile(name: string): boolean;
}
declare class Database extends YamlMulti<Game> {
    get_game(game: string): Game?;
    get_product(game: string, product: string): Product?;
    get_version(game: string, product: string, version: string): ProductVersion?;
    get_profile(game: string, name: string): Profile?;
}

declare class Loader {
    readonly exe: string;
    config: Config;
    database: Database;
    
    // TODO: add more functions
    //upgrade(): boolean;
    //find_game(filename: string): string[];
    //update_protocol() : void;
    //update() : boolean;
    fetch(): string;
    async async_fetch(): string;
}

declare let loader: Loader;